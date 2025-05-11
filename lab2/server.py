import asyncio
import websockets
import json
from Clobber import Clobber

players = []
game = None


async def handle_player(websocket):
    global players, game

    if game is not None:
        await websocket.close()
        return

    players.append(websocket)
    print(f"Gracz {len(players)} dołączył. Obecnie: {len(players)}")

    while len(players) < 2:
        await websocket.send(json.dumps({"type": "wait"}))
        await asyncio.sleep(0.1)

    await asyncio.sleep(0.2)
    try:
        for p in players:
            await p.send(json.dumps({"type": "ping"}))
    except Exception as e:
        return

    if websocket == players[0]:
        asyncio.create_task(run_game())

    await asyncio.Future()


async def run_game():
    global game
    print("Obaj gracze połączeni. Start gry.")
    board = generate_initial_board()
    game = Clobber(board)
    print("Plansza początkowa:")
    game.print_board()

    await broadcast_board()

    try:
        while not game.is_terminal():
            current_player_ws = players[0 if game.current_player == 'B' else 1]

            await current_player_ws.send(json.dumps({
                "type": "your_turn",
                "board": game.board,
                "player": game.current_player
            }))

            move_data = await current_player_ws.recv()
            move = json.loads(move_data)["move"]
            print(f"Gracz {game.current_player} -> {move}")
            game = game.make_move(tuple(map(tuple, move)))
            print("Plansza po ruchu:")
            game.print_board()
            await broadcast_board()

        winner = game.get_opponent()
        print("Gra zakończona. Wygrał:", winner)
        await broadcast({"type": "game_over", "winner": winner})

    except Exception as e:
        print("Błąd:", e)
    finally:
        players.clear()
        game = None


def generate_initial_board(rows=5, cols=6):
    return [['B' if (i + j) % 2 == 0 else 'W' for j in range(cols)] for i in range(rows)]


async def broadcast_board():
    await broadcast({
        "type": "board_update",
        "board": game.board,
        "current": game.current_player
    })


async def broadcast(message):
    for p in players:
        try:
            await p.send(json.dumps(message))
        except:
            pass


async def main():
    async with websockets.serve(handle_player, "localhost", 8765):
        print("Serwer wystartował na ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
