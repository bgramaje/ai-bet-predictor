import soccerdata as sd

fbref = sd.FBref(leagues="ESP-La Liga", seasons=2024)
# print(fbref.__doc__)

laliga_hist = sd.MatchHistory('ESP-La Liga', range(2024,2025))
games = laliga_hist.read_games()
games.sample(5)