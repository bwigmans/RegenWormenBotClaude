
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game_models import Helping, Player, GameState

def test_helping():
    h = Helping(21, 1)
    assert h.number == 21
    assert h.worms == 1
    assert h.face_up == True

def test_player():
    p = Player(0)
    assert p.pid == 0
    assert p.top_helping is None
    h1 = Helping(21,1)
    p.add_helping(h1)
    assert p.top_helping == h1
    h2 = Helping(22,2)
    p.add_helping(h2)
    assert p.top_helping == h2
    removed = p.remove_top_helping()
    assert removed == h2
    assert p.top_helping == h1

def test_gamestate():
    grill = [Helping(21,1), Helping(22,2)]
    grill[0].face_up = True
    grill[1].face_up = False
    players = [Player(0), Player(1)]
    # Give player 1 a top helping
    players[1].add_helping(Helping(23,3))
    state = GameState(grill, players, current_player=0)
    visible = state.visible_grill()
    assert len(visible) == 1
    assert visible[0].number == 21
    others = state.other_top_helpings()
    assert len(others) == 1
    assert others[0][0] == 1
    assert others[0][1].number == 23


if __name__ == "__main__":
    # Run all test functions
    test_functions = [
        test_helping,
        test_player,
        test_gamestate,
    ]

    for test_func in test_functions:
        try:
            test_func()
            print(f"PASS {test_func.__name__}")
        except AssertionError as e:
            print(f"FAIL {test_func.__name__}: {e}")
            raise

    print("\nAll tests passed!")