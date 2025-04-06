import asyncio
import websockets
import json
import argparse
from Clobber import Clobber
from heuristic import heuristic_1, heuristic_2
from minimax import minimax
from minimaxAlphaBeta import minimax_ab

# 🔁 Obsługa heurystyk
heuristics = {
    "h1": heuristic_1,
    "h2": heuristic_2
}

# 🔁 Obsługa algorytmów
algorithms = {
    "minimax": minimax,
    "alphabeta": minimax_ab
}


async def run_agent(depth, heuristic, algo_name):
    uri = "ws://localhost:8765"
    algorithm = algorithms[algo_name]
    heuristic_fn = heuristics[heuristic]

    async with websockets.connect(uri) as websocket:
        try:
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "your_turn":
                    await handle_turn(data, websocket, depth, algorithm, heuristic_fn)
                elif msg_type == "wait":
                    print("Czekam na drugiego gracza...")
                elif msg_type == "board_update":
                    print("Aktualizacja planszy. Ruch gracza:", data.get("current"))
                elif msg_type == "game_over":
                    print("Gra zakończona. Wygrał:", data.get("winner"))
                    break
                elif msg_type == "ping":
                    pass
                else:
                    print("Nieznany typ wiadomości:", data)

        except websockets.exceptions.ConnectionClosedOK:
            print("Połączenie zamknięte (OK)")
        except websockets.exceptions.ConnectionClosedError as e:
            print("Połączenie zerwane z błędem:", e)
        except Exception as e:
            print("Błąd agenta:", e)


async def handle_turn(data, websocket, depth, algorithm, heuristic_fn):
    board = data["board"]
    player = data["player"]
    game = Clobber(board, current_player=player)

    if algorithm == minimax_ab:
        _, move = algorithm(game, depth, float('-inf'), float('inf'), True, heuristic_fn)
    else:
        _, move = algorithm(game, depth, True, heuristic_fn)

    print(f"({player}):", move)
    await websocket.send(json.dumps({"move": move}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3, help="Głębokość przeszukiwania")
    parser.add_argument("--heuristic", choices=["h1", "h2", "h3"], default="h1", help="Heurystyka: h1 lub h2")
    parser.add_argument("--algo", choices=["minimax", "alphabeta"], default="alphabeta",
                        help="Algorytm: minimax lub alphabeta")
    args = parser.parse_args()

    asyncio.run(run_agent(args.depth, args.heuristic, args.algo))
