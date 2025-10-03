from dotenv import load_dotenv
from typing import TypedDict
from enum import Enum
from urllib.parse import quote
import requests
import os
import json
import csv
import sys
import numpy as np

# not actually enforcing any of these but just helpful for readability
class Winner(str, Enum):
	TEAM = "team"
	OPPONENT = "opponent"
	DRAW = "draw"

class Deck(TypedDict):
	cards: list
	supportCards: list # this is just the princess tower troop. named supportCards for consistency with official API

class GameInfo(TypedDict):
	winner: Winner
	team_crowns: int
	opponent_crowns: int
	team_deck: Deck
	opponent_deck: Deck

'''
- MatchupTable Wraps multiple n x n tables, where n is the number of unique cards in Clash Royale (tower troops included) for each table
- The pairwise winrates for each card in the game against each other is calculated in a main table from two separate tables tracking wins and total games for each matchup, respectively

- We use Bayesian Average Winrate with a small prior to calculate pairwise winrates between cards
'''
class MatchupTable:
	def __init__(self, api_key: str, base_url: str):
		self.api_key = api_key
		self.base_url = base_url
		self.headers = { 'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json' }

		self.wins_table: np.ndarray | None = None # shape (n, n) dtype float64
		self.total_games_table: np.ndarray | None = None
		self.winrates_table: np.ndarray | None = None

		self.card_name_to_index: dict[str, int] = {}
		self.index_to_card_name: dict[int, str] = {}

		self.gamesProcessed = 0

		self.__init_tables()

	def __init_tables(self):
		url = self.base_url + '/cards'

		# getting cards from official api to future proof. clash adds new cards like every month
		# response contains json string of 2 arrays, 'items' (troops, buildings, spells) and 'supportItems' (princess tower troops)
		response = requests.get(url, headers=self.headers)
		if response.status_code != 200:
			raise Exception('[__init_tables(): Error making request]')

		response_dict = json.loads(response.text)

		# extracting these arrays, each of these arrays contains a dictionary with information of that card/tower troop
		cards = response_dict["items"]
		tower_troops = response_dict["supportItems"]
		all_cards = cards + tower_troops
		n = len(all_cards)

		# build mappings
		for index, card in enumerate(all_cards):
			name = card['name'].lower()
			self.card_name_to_index[name] = index
			self.index_to_card_name[index] = name

		self.wins_table = np.zeros((n, n), dtype=np.float64)
		self.total_games_table = np.zeros((n, n), dtype=np.float64)
		self.winrates_table = np.zeros((n, n), dtype=np.float64)
	
	def print_wins_table(self) -> None:
		print(self.wins_table)

	def _indices_for_deck(self, deck: Deck) -> np.ndarray:
		# return numpy array of indices for a deck's cards + support troop, filtered to known names
		names = deck['cards'] + deck['supportCards']
		idxs = [self.card_name_to_index[name] for name in names if name in self.card_name_to_index]
		return np.unique(np.array(idxs, dtype=np.int32))

	# Update table values according to the above comment block
	def process_game(self, game_info: GameInfo):
		if self.wins_table is None or self.total_games_table is None:
			return
		
		if game_info['winner'] == Winner.DRAW:
			return
		
		team_idx = self._indices_for_deck(game_info['team_deck'])
		opp_idx = self._indices_for_deck(game_info['opponent_deck'])

		if team_idx.size == 0 or opp_idx.size == 0:
			return
		
		if game_info['winner'] == Winner.TEAM:
			# team beats opponent: + 1 for team
			self.wins_table[np.ix_(team_idx, opp_idx)] += 1.0
			#self.wins_table[np.ix_(opp_idx, team_idx)] -= 1.0
		elif game_info['winner'] == Winner.OPPONENT:
			# opponent beats team: + 1 for opponent
			self.wins_table[np.ix_(opp_idx, team_idx)] += 1.0
			#self.wins_table[np.ix_(team_idx, opp_idx)] -= 1.0

		# update total games played regardless of winner to record the matchup
		self.total_games_table[np.ix_(team_idx, opp_idx)] += 1.0
		self.total_games_table[np.ix_(opp_idx, team_idx)] += 1.0

		self.gamesProcessed += 1

	def process_player_recent_ranked_games(self, player_tag: str):
		url = self.base_url + f'/players/{quote(player_tag.upper())}/battlelog'
		response = requests.get(url, headers=self.headers)
		if response.status_code != 200:
			raise Exception('[process_player_recent_ranked_games(): Error making request]')

		games_list = json.loads(response.text)
		for game in games_list:
			# only process ranked games
			if game['type'] == 'pathOfLegend': # clash royale api still uses pathOfLegend when referring to ranked
				# get winner
				team_crowns = game['team'][0]['crowns']
				opponent_crowns = game['opponent'][0]['crowns']

				winner = Winner.TEAM if team_crowns > opponent_crowns else (Winner.OPPONENT if opponent_crowns > team_crowns else Winner.DRAW)

				team_cards_data = game['team'][0]['cards']
				team_cards = [c['name'].lower() for c in team_cards_data if c['name'].lower() in self.card_name_to_index]
				team_support = [game['team'][0]['supportCards'][0]['name'].lower()]
				team_deck: Deck = { 'cards': team_cards, 'supportCards': team_support }

				opponent_cards_data = game['opponent'][0]['cards']
				opponent_cards = [c['name'].lower() for c in opponent_cards_data if c['name'].lower() in self.card_name_to_index]
				opponent_support = [game['opponent'][0]['supportCards'][0]['name'].lower()]
				opponent_deck: Deck = { 'cards': opponent_cards, 'supportCards': opponent_support }

				game_info: GameInfo = { 'winner': winner, 'team_crowns': team_crowns, 'opponent_crowns': opponent_crowns, 'team_deck': team_deck, 'opponent_deck': opponent_deck }

				self.process_game(game_info)

	def get_top_n_ranked_player_tags_by_season(self, season: str, n: int = 200) -> list[str]:
		url = self.base_url + f'/locations/global/pathoflegend/{season}/rankings/players?limit={n}'

		response = requests.get(url, headers=self.headers)
		if response.status_code != 200:
			raise Exception('[get_top_n_ranked_player_tags_by_season(): Error making request]')
			
		return [p['tag'] for p in json.loads(response.text)['items']]

	# big boy function that gets top n ranked players for a certain season and counts the card to card matchup results in the matchup table
	def count_top_n_ranked_player_games(self, season: str, n: int = 200):
		top_player_tags = self.get_top_n_ranked_player_tags_by_season(season, n)
		
		for index, top_player_tag in enumerate(top_player_tags, 1):
			print(f'Processing recent ranked games for player tag {top_player_tag} ({index}/{len(top_player_tags)})...')

			try:
				self.process_player_recent_ranked_games(top_player_tag)
			except Exception as e:
				print('Something went wrong: {e}\nSaving current MatchupTable and exiting...')
				self.save_to_csv('matchup_table.txt')
				return

			print('Success!')

		print(f'Processed {self.gamesProcessed} games.')

	def calculate_winrates(self):
		rows = self.wins_table.shape[0]
		cols = self.wins_table.shape[1]

		for i in range(rows):
			for j in range(cols):
				wins = self.wins_table[i, j]
				total_games = self.total_games_table[i, j]
				self.winrates_table[i, j] = self.winrate_function(wins, total_games)

	def winrate_function(self, wins, total_games, smoothing=1.0):
		# n smoothing adds n win and n loss
		return (wins + smoothing) / (total_games + (2 * smoothing))
	
	# saves current card name <-> index mappings + table values to csv file for persistence
	def save_to_csv(self, filename: str):
		if self.wins_table is None or self.total_games_table is None:
			raise RuntimeError("Table is not initialized")

		with open(filename, 'w', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)

			# Writing card name <-> index mappings
			for key, value in self.card_name_to_index.items():
				csv_writer.writerow([key, value])
		
			csv_writer.writerow([]) # empty row divider needed for load_from_csv()

			# Writing wins table values
			for i in range(self.wins_table.shape[0]):
				row = [i] + self.wins_table[i].tolist()
				csv_writer.writerow(row)
			
			csv_writer.writerow([]) # empty row divider needed for load_from_csv()

			# Writing total games played table values
			for i in range(self.total_games_table.shape[0]):
				row = [i] + self.total_games_table[i].tolist()
				csv_writer.writerow(row)

			csv_writer.writerow([]) # empty row divider needed for load_from_csv()

			# Writing winrate table values
			for i in range(self.winrates_table.shape[0]):
				row = [i] + self.winrates_table[i].tolist()
				csv_writer.writerow(row)

		print(f'Successfully written to {filename}')

	# load table and mappings from existing csv
	# note: this expects a very rigid structure defined by save_to_csv()
	def load_from_csv(self, filename: str):
		self.card_name_to_index.clear()
		self.index_to_card_name.clear()
		wins_rows: list[list[float]] = []
		total_games_rows: list[list[float]] = []
		winrates_rows: list[list[float]] = []

		with open(filename, 'r', newline='') as csvfile:
			csv_reader = csv.reader(csvfile)

			# 1 divider hit means we're now reading wins table values, 2 means we're reading total games table values
			dividers_hit = 0

			for row in csv_reader:
				if len(row) == 0:
					dividers_hit += 1
					continue

				if dividers_hit == 0:
					# reading mappings
					name = row[0]
					index = int(row[1])
					self.card_name_to_index[name] = index
					self.index_to_card_name[index] = name

				if dividers_hit == 1:
					# reading wins table values
					# row: [index, v0, v1, ...]
					wins_rows.append([float(x) for x in row[1:]])
				
				# reading total games
				if dividers_hit == 2:
					total_games_rows.append([float(x) for x in row[1:]])
				
				# reading winrates
				if dividers_hit == 3:
					winrates_rows.append([float(x) for x in row[1:]])


		# rebuild numpy tables
		self.wins_table = np.array(wins_rows, dtype=np.float64)
		self.total_games_table = np.array(total_games_rows, dtype=np.float64)
		self.winrates_table = np.array(winrates_rows, dtype=np.float64)

		print(f'Successfully loaded from {filename}')

	def get_winrates_table(self) -> np.ndarray | None:
		return self.winrates_table;

	def get_card_name_to_index_dict(self) -> dict[str, int]:
		return self.card_name_to_index
	
	def get_index_to_card_name_dict(self) -> dict[int, str]:
		return self.index_to_card_name



def main():
	load_dotenv()

	api_key = os.getenv("API_KEY")
	base_url = os.getenv("API_BASE_URL")

	mt = MatchupTable(api_key=api_key, base_url=base_url)
	mt.count_top_n_ranked_player_games('2025-08')
	mt.calculate_winrates()
	mt.save_to_csv('card_matchups_from_recent_ranked_games_of_08_2025_top_200_on_10_02_2025.csv')

if __name__ == "__main__":
	main()