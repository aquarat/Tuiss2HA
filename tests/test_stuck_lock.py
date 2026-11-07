"""Regression tests for issue #144: the busy lock staying held after a failed operation.

The blind's real BLE stack is replaced by FakeClient. Nothing here talks to hardware.
Every wait is guarded by LIMIT, so a lock that is never released fails the test
instead of hanging the run.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.tuiss2ha import hub as hub_module
from custom_components.tuiss2ha import timers as timers_module
from custom_components.tuiss2ha.const import ConnectionTimeout, DeviceNotFound
from custom_components.tuiss2ha.cover import HomeAssistantError, STATE_CLOSING, Tuiss
from custom_components.tuiss2ha.hub import TuissBlind

LIMIT = 2.0  # seconds; far longer than any fake operation takes
SHORT = 0.02  # replacement for the BLE backstop timeouts under test


class Stall:
    """An async callable that blocks until released, like a radio that never answers."""

    def __init__(self, result=None):
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        self.result = result

    async def __call__(self, *args, **kwargs):
        self.entered.set()
        await self.released.wait()
        return self.result


class FakeClient:
    """Stands in for BleakClientWithServiceCache."""

    def __init__(self):
        self.is_connected = True
        self.write_gatt_char = AsyncMock()
        self.start_notify = AsyncMock()
        self.stop_notify = AsyncMock()
        self.disconnect = AsyncMock(side_effect=self._disconnected)

    async def _disconnected(self):
        self.is_connected = False


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def blind(mock_hass, client):
    """A real TuissBlind whose connection attempts return the fake client."""
    device = MagicMock()
    device.name = "TS2900"
    with (
        patch(
            "custom_components.tuiss2ha.hub.bluetooth.async_ble_device_from_address",
            return_value=device,
        ),
        patch(
            "custom_components.tuiss2ha.hub.establish_connection",
            AsyncMock(return_value=client),
        ),
        patch("custom_components.tuiss2ha.hub.TIMEOUT_SECONDS", 0.5),
    ):
        hub = MagicMock()
        mock_hass.async_create_task = asyncio.create_task
        mock_hass.loop = MagicMock()
        hub._hass = mock_hass
        tb = TuissBlind("AA:BB:CC:DD:EE:FF", "Test Blind", hub)
        tb._restart_attempts = 1
        tb._current_cover_position = 50.0
        yield tb


def close_blind(tb):
    """Start the move an automation's cover.close_cover would."""
    return asyncio.create_task(
        tb.async_move_cover(
            movement_direction=-1, target_position=100, skip_battery_check=True
        )
    )


async def cancelled(task):
    """Cancel a task the way HA cancels a service call when its script stops."""
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, LIMIT)


async def finished(awaitable):
    return await asyncio.wait_for(awaitable, LIMIT)


def assert_idle(tb):
    assert tb._locked is False, "lock still held"
    assert tb._moving == 0, "still reported as moving"


async def assert_usable(tb):
    """The test that matters to the owner: the next command gets through."""
    await finished(tb.get_blind_position())
    assert_idle(tb)


def notify_position_when_subscribed(tb, client, position=75):
    """Make the fake blind answer a position query, so reads finish at once."""

    async def start_notify(char, callback):
        data = bytearray([255, 1, 2, 3, 209, 2, 99, position * 10 % 256, position * 10 // 256, 0, 0])
        await callback(char, data)

    client.start_notify.side_effect = start_notify


# --- The report in issue #144 ------------------------------------------------


@pytest.mark.asyncio
async def test_issue_144_cancelled_while_connecting_does_not_stick(blind, client):
    """Close starts, the connection never completes, the caller goes away.

    Before the fix this leaves exactly what the report shows: Lock Status locked,
    Connection Status off, the cover "closing", and every later call "busy".
    """
    stall = Stall()
    with patch("custom_components.tuiss2ha.hub.establish_connection", stall):
        task = close_blind(blind)
        await finished(stall.entered.wait())
        assert blind._locked is True
        assert blind._moving == -1
        await cancelled(task)

    assert_idle(blind)
    cover = Tuiss(blind, MagicMock(options={}))
    assert cover.state != STATE_CLOSING

    notify_position_when_subscribed(blind, client)
    await assert_usable(blind)


# --- Cancellation on every phase of a move -----------------------------------


@pytest.mark.asyncio
async def test_cancel_during_connection_retry_backoff_releases(blind):
    """Cancelled between attempts rather than inside one."""
    sleep = Stall()
    blind._restart_attempts = 2
    with (
        patch(
            "custom_components.tuiss2ha.hub.establish_connection",
            AsyncMock(side_effect=hub_module.BleakError("Disconnected")),
        ),
        patch("custom_components.tuiss2ha.hub.asyncio.sleep", sleep),
    ):
        task = close_blind(blind)
        await finished(sleep.entered.wait())
        await cancelled(task)
    assert_idle(blind)


@pytest.mark.asyncio
async def test_cancel_during_battery_check_releases(blind, client):
    client.start_notify = Stall()
    blind._battery_check_days = 1
    task = asyncio.create_task(
        blind.async_move_cover(movement_direction=-1, target_position=100)
    )
    await asyncio.sleep(SHORT)
    await cancelled(task)
    assert_idle(blind)


@pytest.mark.asyncio
async def test_cancel_during_move_command_releases(blind, client):
    stall = Stall()
    client.start_notify = stall
    task = close_blind(blind)
    await finished(stall.entered.wait())
    await cancelled(task)
    assert_idle(blind)


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_motor_clears_moving(blind, client):
    """The lock was already released here; the cover stayed "closing" for ever."""
    task = close_blind(blind)
    await finished(_until(lambda: client.write_gatt_char.call_count >= 3))
    await cancelled(task)
    assert_idle(blind)
    assert client.disconnect.await_count == 1


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_earlier_post_move_query(blind):
    """A cancellation must not be swallowed while the old query is being stopped."""

    old_query = asyncio.create_task(ignores_first_cancel())
    await asyncio.sleep(0)
    blind._post_move_task = old_query

    task = close_blind(blind)
    await asyncio.sleep(SHORT)
    await cancelled(task)
    assert_idle(blind)
    old_query.cancel()


async def _until(condition):
    while not condition():
        await asyncio.sleep(0.001)


async def ignores_first_cancel():
    """A background task that takes a while to wind down when cancelled."""
    try:
        await asyncio.sleep(LIMIT * 2)
    except asyncio.CancelledError:
        await asyncio.sleep(LIMIT * 2)


# --- Failures that were already handled stay handled -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [DeviceNotFound("gone"), ConnectionTimeout("failed")])
async def test_connection_failure_releases_and_reraises(blind, error):
    blind.attempt_connection = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await finished(close_blind(blind))
    assert_idle(blind)


@pytest.mark.asyncio
async def test_connection_error_on_every_attempt_releases(blind, client):
    with patch(
        "custom_components.tuiss2ha.hub.establish_connection",
        AsyncMock(side_effect=hub_module.BleakError("Disconnected")),
    ):
        with pytest.raises(ConnectionTimeout):
            await finished(close_blind(blind))
    assert_idle(blind)
    assert "Disconnected" in blind._last_connection_error


@pytest.mark.asyncio
async def test_failed_move_command_releases(blind, client):
    client.start_notify.side_effect = RuntimeError("radio fault")
    await finished(close_blind(blind))
    assert_idle(blind)


@pytest.mark.asyncio
async def test_read_cancelled_releases(blind, client):
    stall = Stall()
    client.start_notify = stall
    task = asyncio.create_task(blind.get_blind_position())
    await finished(stall.entered.wait())
    await cancelled(task)
    assert_idle(blind)


# --- A radio that never answers ----------------------------------------------


@pytest.fixture
def short_backstops():
    with (
        patch.object(hub_module, "BLE_CONNECT_TIMEOUT", SHORT, create=True),
        patch.object(hub_module, "BLE_GATT_TIMEOUT", SHORT, create=True),
        patch.object(hub_module, "BLE_DISCONNECT_TIMEOUT", SHORT, create=True),
    ):
        yield


@pytest.mark.asyncio
async def test_stalled_connect_times_out_and_releases(blind, short_backstops):
    with patch("custom_components.tuiss2ha.hub.establish_connection", Stall()):
        with pytest.raises(ConnectionTimeout):
            await finished(close_blind(blind))
    assert_idle(blind)
    assert blind._last_connection_error, "the failure should be visible on the sensor"


@pytest.mark.asyncio
async def test_stalled_handshake_write_releases(blind, client, short_backstops):
    """Connected, then the first GATT write never completes."""
    client.write_gatt_char = Stall()
    await finished(close_blind(blind))
    assert_idle(blind)


@pytest.mark.asyncio
async def test_stalled_notify_on_read_releases(blind, client, short_backstops):
    client.start_notify = Stall()
    with pytest.raises(HomeAssistantError):
        await finished(blind.get_blind_position())
    assert_idle(blind)


@pytest.mark.asyncio
async def test_stalled_write_on_stop_surfaces_as_handled_error(blind, client, short_backstops):
    """async_stop_cover handles RuntimeError; a bare TimeoutError would escape it."""
    blind._moving = -1
    blind._locked = True
    client.write_gatt_char = Stall()
    blind._client = client
    with pytest.raises(RuntimeError):
        await finished(blind.stop())
    assert_idle(blind)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["stop_notify", "disconnect"])
async def test_stalled_teardown_releases(blind, client, short_backstops, step):
    notify_position_when_subscribed(blind, client)
    setattr(client, step, Stall())
    await finished(blind.get_blind_position())
    assert_idle(blind)


@pytest.mark.asyncio
async def test_stalled_disconnect_does_not_leave_a_wedged_client(blind, client, short_backstops):
    """A client that would not disconnect must not be reused as "already connected"."""
    notify_position_when_subscribed(blind, client)
    client.disconnect = Stall()
    await finished(blind.get_blind_position())
    assert blind._client is None


@pytest.mark.asyncio
async def test_stubborn_post_move_query_cannot_hold_up_a_move(blind, client, short_backstops):
    old_query = asyncio.create_task(ignores_first_cancel())
    await asyncio.sleep(0)
    blind._post_move_task = old_query

    task = close_blind(blind)
    await finished(_until(lambda: client.write_gatt_char.call_count >= 3))
    blind._stopped_event.set()
    await finished(task)
    assert_idle(blind)
    old_query.cancel()


@pytest.mark.asyncio
async def test_stalled_notify_stop_while_adding_timer_releases(blind, client):
    async def reply_with_timer_id(char, callback):
        await callback(char, bytearray([255, 1, 2, 3, 214, 0, 5]))

    client.start_notify.side_effect = reply_with_timer_id
    client.stop_notify = Stall()
    with (
        patch.object(timers_module, "BLE_GATT_TIMEOUT", SHORT, create=True),
        patch.object(hub_module, "BLE_DISCONNECT_TIMEOUT", SHORT, create=True),
        pytest.raises(TimeoutError),
    ):
        await finished(blind.async_add_timer(["mon"], "08:00", 50.0))
    assert_idle(blind)


@pytest.mark.asyncio
async def test_stalled_disconnect_while_deleting_timers_releases(blind, client):
    client.disconnect = Stall()
    with (
        patch.object(timers_module, "BLE_DISCONNECT_TIMEOUT", SHORT, create=True),
        patch.object(hub_module, "BLE_DISCONNECT_TIMEOUT", SHORT, create=True),
        pytest.raises(TimeoutError),
    ):
        await finished(blind.delete_all_timers())
    assert_idle(blind)


# --- Link lost mid-move: blocking semantics kept, lock still released --------


@pytest.mark.asyncio
async def test_link_lost_mid_move_blocks_to_timeout_then_releases(blind, client):
    """The call keeps blocking: a lost link says nothing about the motor having stopped."""
    with patch("custom_components.tuiss2ha.hub.TIMEOUT_SECONDS", 0.1):
        task = close_blind(blind)
        await finished(_until(lambda: client.write_gatt_char.call_count >= 3))
        client.is_connected = False
        await asyncio.sleep(0.03)
        assert not task.done(), "must not return before the motor could have stopped"
        assert blind._locked is True
        await finished(task)
    assert_idle(blind)
    blind._post_move_task.cancel()


# --- Stop and force unlock against an operation that has gone stale ----------


@pytest.mark.asyncio
async def test_stop_recovers_a_blocked_move(blind, client):
    stall = Stall()
    with patch("custom_components.tuiss2ha.hub.establish_connection", stall):
        task = close_blind(blind)
        await finished(stall.entered.wait())
        with patch(
            "custom_components.tuiss2ha.hub.establish_connection",
            AsyncMock(return_value=client),
        ):
            await finished(blind.stop())
        assert_idle(blind)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_move_does_not_resume_under_a_newer_operation(blind, client):
    """A move released by force unlock must abort, not send its command later."""
    stale_connect = Stall(result=client)
    connects = iter([stale_connect, AsyncMock(return_value=client)])

    async def establish(**kwargs):
        return await next(connects)(**kwargs)

    with patch("custom_components.tuiss2ha.hub.establish_connection", establish):
        stale = close_blind(blind)
        await finished(stale_connect.entered.wait())

        # What the force_unlock service does to the blind.
        blind._locked = False
        blind._moving = 0

        newer = asyncio.create_task(
            blind.async_move_cover(
                movement_direction=1, target_position=0, skip_battery_check=True
            )
        )
        await finished(_until(lambda: client.start_notify.await_count == 1))

        stale_connect.released.set()
        await asyncio.sleep(SHORT)

        assert blind._locked is True
        assert blind._moving == 1
        assert blind._desired_position == 100, "stale target overwrote the newer one"
        assert client.start_notify.await_count == 1, "stale move sent its command"

        blind._stopped_event.set()
        await finished(asyncio.gather(stale, newer))


@pytest.mark.asyncio
async def test_stale_read_does_not_release_a_newer_lock(blind, client):
    stall = Stall()
    client.start_notify = stall
    stale = asyncio.create_task(blind.get_blind_position())
    await finished(stall.entered.wait())

    # What the force_unlock service does to the blind.
    blind._locked = False

    newer = asyncio.create_task(blind.get_blind_position())
    await asyncio.sleep(SHORT)
    assert blind._locked is True

    stale.cancel()
    await asyncio.gather(stale, return_exceptions=True)

    assert blind._locked is True, "stale read released the newer read's lock"
    client.disconnect.assert_not_awaited()

    newer.cancel()
    await asyncio.gather(newer, return_exceptions=True)
    assert_idle(blind)


@pytest.mark.asyncio
async def test_stale_move_timing_out_leaves_newer_operation_alone(blind, client):
    with patch("custom_components.tuiss2ha.hub.TIMEOUT_SECONDS", 0.05):
        stale = close_blind(blind)
        await finished(_until(lambda: client.write_gatt_char.call_count >= 3))

        # What the force_unlock service does to the blind.
        blind._locked = False
        blind._moving = 0
        blind._current_cover_position = 50.0

        stall = Stall()
        client.start_notify = stall
        newer = asyncio.create_task(
            blind.async_move_cover(
                movement_direction=1, target_position=0, skip_battery_check=True
            )
        )
        await finished(stall.entered.wait())
        await finished(stale)

    assert blind._locked is True
    assert blind._moving == 1, "stale timeout cleared the newer move"
    assert blind._current_cover_position == 50.0, "stale timeout wrote its target"

    newer.cancel()
    await asyncio.gather(newer, return_exceptions=True)


@pytest.mark.asyncio
async def test_force_unlock_also_clears_moving(blind):
    blind._locked = True
    blind._moving = -1
    blind.force_unlock()
    assert_idle(blind)


# --- Release ordering --------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_held_until_teardown_finishes(blind, client):
    """A command let in during teardown would find a link about to be closed."""
    notify_position_when_subscribed(blind, client)
    stall = Stall()
    client.disconnect = stall
    read = asyncio.create_task(blind.get_blind_position())
    await finished(stall.entered.wait())

    with pytest.raises(HomeAssistantError) as exc:
        await finished(blind.async_move_cover(movement_direction=1, target_position=0))
    assert exc.value.translation_key == "device_locked"

    stall.released.set()
    await finished(read)
    assert_idle(blind)


@pytest.mark.asyncio
async def test_second_cancel_during_teardown_still_releases(blind, client):
    stall = Stall()
    client.start_notify = stall
    teardown = Stall()
    client.disconnect = teardown

    task = close_blind(blind)
    await finished(stall.entered.wait())
    task.cancel()
    await finished(teardown.entered.wait())
    await cancelled(task)
    assert_idle(blind)


@pytest.mark.asyncio
async def test_teardown_error_does_not_mask_or_stick(blind, client):
    notify_position_when_subscribed(blind, client)
    client.disconnect = AsyncMock(side_effect=ValueError("unexpected"))
    await finished(blind.get_blind_position())
    assert_idle(blind)
