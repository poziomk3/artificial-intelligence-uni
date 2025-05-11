import asyncio
import websockets
import json
import argparse
from heuristic import *
from minimaxAlphaBeta import minimax_ab

HEURISTIC_DICT = {name: func for name, func in [
    ("heuristic_1", heuristic_1),
    ("heuristic_2", heuristic_2),
    ("heuristic_3", heuristic_3),
    ("heuristic_4", heuristic_4),
    ("heuristic_5", heuristic_5),
    ("heuristic_6", heuristic_6),
]}


async def run_agent(depth, heuristic_name):
    uri = "ws://localhost:8765"
    heuristic_fn = HEURISTIC_DICT[heuristic_name]

    async with websockets.connect(uri) as websocket:
        print(f"[Agent] Connected to server | Heuristic: {heuristic_name} | Depth: {depth}")

        try:
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "your_turn":
                    await handle_turn(data, websocket, depth, heuristic_fn)
                elif msg_type == "wait":
                    # print("Czekam na drugiego gracza...")
                    pass
                elif msg_type == "board_update":
                    # print("Aktualizacja planszy. Ruch gracza:", data.get("current"))
                    pass
                elif msg_type == "game_over":
                    # print("Gra zakończona. Wygrał:", data.get("winner"))
                    break
                elif msg_type == "ping":
                    pass

        except websockets.exceptions.ConnectionClosed:
            print("Połączenie z serwerem zamknięte.")


async def handle_turn(data, websocket, depth, heuristic_fn):
    board = data["board"]
    player = data["player"]
    game = Clobber(board, current_player=player)

    maximizing = player == 'B'
    nodes_visited = [0]

    _, move = minimax_ab(game, depth, float('-inf'), float('inf'), maximizing, heuristic_fn, nodes_visited)

    print(f"[{player}] Ruch: {move} | Odwiedzone węzły: {nodes_visited[0]}")
    await websocket.send(json.dumps({"move": move}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3, help="Głębokość przeszukiwania")
    parser.add_argument("--heuristic", choices=HEURISTIC_DICT.keys(), required=True)
    args = parser.parse_args()

    asyncio.run(run_agent(args.depth, args.heuristic))
