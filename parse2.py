import json
import time

import pandas as pd
import csv
import numpy as np
from math import pi
import time

class Killparser:
    """
    PARSES TICKS LEADING UP TO A KILL AND SPLITS THEM IN CSVs
    """

    def __init__(self):
        self.x = {}
        self.n_rounds = None
        self.MatchId = None

    def read_json(self,file_path):
        with open(file_path,encoding='UTF-8') as f:
            x = json.load(f)
            self.x = x
            self.n_rounds = len(self.x["GameRounds"])
            self.MatchId = self.x["MatchId"]

    def get_players_ids(self):
        players = []
        try:
            for t in range(2):
                if t == 0:
                    team_index = "T"
                else:
                    team_index = "CT"
                for i in range(5):
                    players.append((self.x["GameRounds"][1]["Frames"][0][team_index]["Players"][i]["Name"],self.x["GameRounds"][1]["Frames"][1][team_index]["Players"][i]["SteamId"]))
        except Exception as e:
            print(e)
        return players


    def get_pos_player(self, steamid, main_folder):
        """
        :param steamid:
        :return: None
        """
        # Write headers for file
        with open(f"{main_folder}{self.MatchId}.csv",'w',newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(['sus','ActiveWeapon', 'AreaId', 'AreaName', 'Armor', 'DistToBombsiteA',
                                 'DistToBombsiteB', 'EquipmentValue', 'HasDefuse', 'HasHelmet', 'Hp',
                                 'Inventory', 'IsAirborne', 'IsAlive', 'IsDucking', 'IsFlashed',
                                 'IsScoped', 'IsWalking', 'Money', 'Name', 'SteamId', 'TICK',
                                 'TotalUtility', 'ViewX', 'ViewY', 'X', 'Y', 'Z'])

        # Get enemy ids
        t_players = []
        ct_players = []
        for p in range(5):
            t_player = self.x["GameRounds"][12]["Frames"][3]["T"]["Players"][p]['SteamId']
            t_players.append(t_player)
            ct_player = self.x["GameRounds"][12]["Frames"][3]["CT"]["Players"][p]['SteamId']
            ct_players.append(ct_player)

        enemies = []
        if int(steamid) in ct_players:
            enemies = t_players
            print("SUSPECT IS CT")
        elif int(steamid) in t_players:
            enemies = ct_players
            print("SUSPECT IS T")


        # Get every tick (could be optimized by only getting the correct ticks)
        for rounds in range(self.n_rounds):
            for frames in range(len(self.x["GameRounds"][rounds]["Frames"])):
                for team in range(2):
                    for player in range(4):
                        if team == 1:
                            team_index = "T"
                        else:
                            team_index = "CT"
                        try:
                            # print(self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]["SteamId"]," IN  ",enemies)
                            if self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]["SteamId"] in enemies:

                                dikt = self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]
                                dikt["TICK"] = self.x["GameRounds"][rounds]["Frames"][frames]["Tick"]

                                df = pd.DataFrame.from_records(dikt)

                                df.to_csv(f"{main_folder}{self.MatchId}.csv", mode='a',header=False)
                        except Exception as e:
                            pass


    def get_kills_csv(self,kill_folder):
        """
        Gets all the kills in the match
        """
        # Headers
        with open(f"{kill_folder}{self.MatchId}.csv",'w',newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(['sus','AssistedFlash', 'AssisterAreaId', 'AssisterAreaName', 'AssisterName',
                                   'AssisterSide', 'AssisterSteamId', 'AssisterTeam', 'AssisterX',
                                   'AssisterY', 'AssisterZ', 'AttackerAreaId', 'AttackerAreaName',
                                   'AttackerBlind', 'AttackerName', 'AttackerSide', 'AttackerSteamId',
                                   'AttackerTeam', 'AttackerViewX', 'AttackerViewY', 'AttackerX',
                                   'AttackerY', 'AttackerZ', 'Distance', 'IsFirstKill', 'IsFlashed',
                                   'IsHeadshot', 'IsTrade', 'IsWallbang', 'NoScope', 'PenetratedObjects',
                                   'PlayerTradedName', 'PlayerTradedSteamId', 'PlayerTradedTeam', 'Second',
                                   'ThruSmoke', 'TICK', 'VictimAreaId', 'VictimAreaName', 'VictimName',
                                   'VictimSide', 'VictimSteamId', 'VictimTeam', 'VictimViewX',
                                   'VictimViewY', 'VictimX', 'VictimY', 'VictimZ', 'Weapon'])

        for rounds in range(self.n_rounds):
            try:
                for k in range(len(self.x["GameRounds"][rounds]["Kills"])):
                    df = pd.DataFrame.from_records(self.x["GameRounds"][rounds]["Kills"][k],index=[0])
                    #df.to_sql(name='kill2', con=self.dbConnection, index=False, if_exists='append')
                    df.to_csv(f"{kill_folder}{self.MatchId}.csv", mode='a', header=False)
            except Exception as e:
                print(e)

    def split_df_by_kill(self,mainfile,killfile,steamid,out_folder,write_to_csv=False):
        """
        :param attacker_name:
        :return: List of dfs consisting of ticks up to the kill
        """

        df1 = pd.read_csv(mainfile)
        df2 = pd.read_csv(killfile)

        n_files = len([name for name in os.listdir(r'D:\Users\emill\csgocheaters\enemy/')])
        cnt = 0

        suspect = steamid

        for inx in range(len(df2)):
            single_kill = df2.iloc[inx]

            if single_kill["AttackerSteamId"] == suspect:
                victim_this_kill = single_kill["VictimSteamId"]
                tick_this_kill = single_kill["TICK"]

                df_victim = df1[df1["SteamId"] == victim_this_kill]
                df_victim = df_victim[df_victim["TICK"] <= tick_this_kill]
                # df_victim = df_victim[["X", "Y", "Z", "SteamId", "TICK", "Name"]]

                df_killer = df1[df1["SteamId"] == np.int(suspect)]
                joined_df = df_killer.merge(df_victim, how='left', on="TICK")

                joined_df = joined_df[joined_df["TICK"] < tick_this_kill]
                joined_df = joined_df.fillna(method='ffill')
                # Trig
                """joined_df['ViewX'] = (joined_df['ViewX_x'] + 180) % 360 - 180
                joined_df['ViewY'] = (joined_df['ViewY_x'] + 180) % 360 - 180

                x1 = joined_df["X_x"]
                y1 = joined_df["Y_x"]
                z1 = joined_df["Z_x"]

                x2 = joined_df["X_y"]
                y2 = joined_df["Y_y"]
                z2 = joined_df["Z_y"]

                xdif = np.array(x2 - x1)
                ydif = np.array(y2 - y1)
                zdif = np.array(z2 - z1)

                rightLeft = np.arctan2(ydif, xdif) * (180 / pi)
                hypot3D = np.sqrt(zdif ** 2 + xdif ** 2 + ydif ** 2)
                upDown = (-np.arcsin(zdif / hypot3D)) * (180 / pi)

                joined_df["X_Off_By_Degrees"] = rightLeft - joined_df["ViewX_x"]
                joined_df["Y_Off_By_Degrees"] = upDown - joined_df["ViewY_x"]

                single_kill = single_kill.rename(
                    {"AttackerViewX": "ViewX", "AttackerViewY": "ViewY", "VictimX": "X_y", "VictimY": "Y_y",
                     "VictimZ": "Z_y", "AttackerX": "X_x", "AttackerY": "Y_x", "AttackerZ": "Z_x"})
                # last_joining_row = single_kill[["X_y", "Y_y", "Z_y", "X_x", "Y_x", "Z_x", "ViewX", "ViewY"]]

                # Also do trig for the last row
                last_joining_row = single_kill
                last_joining_row['ViewX_x'] = (last_joining_row['ViewX_x'] + 180) % 360 - 180
                last_joining_row['ViewY_x'] = (last_joining_row['ViewY_x'] + 180) % 360 - 180

                x1 = last_joining_row["X_x"]
                y1 = last_joining_row["Y_x"]
                z1 = last_joining_row["Z_x"]

                x2 = last_joining_row["X_y"]
                y2 = last_joining_row["Y_y"]
                z2 = last_joining_row["Z_y"]

                xdif = np.array(x2 - x1)
                ydif = np.array(y2 - y1)
                zdif = np.array(z2 - z1)

                rightLeft = np.arctan2(ydif, xdif) * (180 / pi)
                hypot3D = np.sqrt(zdif ** 2 + xdif ** 2 + ydif ** 2)
                upDown = (-np.arcsin(zdif / hypot3D)) * (180 / pi)

                last_joining_row["X_Off_By_Degrees"] = rightLeft - last_joining_row["ViewX"]
                last_joining_row["Y_Off_By_Degrees"] = upDown - last_joining_row["ViewY"]"""

                joined_df = joined_df.append(single_kill)
                joined_df = joined_df.fillna(method='ffill')
                cnt += 1
                if len(joined_df) > 3000:
                    joined_df = joined_df.iloc[len(joined_df) - 3000:]
                joined_df.to_csv(f"{out_folder}/{n_files + cnt}.csv")



def main(json_name, steamid):
    """
    Takes in a parsed json file and a player name. Creates CSV files from each kill, having
    the ticks before the kill happened.
    :param json_name:
    :param steamid:
    :return:
    """
    started = time.time()

    datamover = Killparser()
    print("Killparser",time.time()-started)
    datamover.read_json(f"D:/Users/emill/shitter/a/{json_name}")
    print("read_json", time.time() - started)
    try:
        datamover.get_pos_player(f"{steamid}",f"D:/Users/emill/csgocheaters/games/")
    except Exception as e:
        print("FAILED ON:",e)
    print("get_pos_player", time.time() - started)
    datamover.get_kills_csv(r'D:\Users\emill\csgocheaters\kills/')
    print("get_kills_csv", time.time() - started)

    json_name_without_end = json_name.replace(".json","")
    mainfile = f'D:/Users/emill/csgocheaters/games/{json_name_without_end}.csv'
    killfile = f'D:/Users/emill/csgocheaters/kills/{json_name_without_end}.csv'
    out_folder = r"D:\Users\emill\csgocheaters\enemy/"
    datamover.split_df_by_kill(mainfile, killfile, steamid, out_folder, write_to_csv=True)
    print("split_df_by_kill", time.time() - started)


def get_names(json_name):
    datamover = Killparser()
    datamover.read_json(f"D:/Users/emill/shitter/a/{json_name}")
    players = datamover.get_players_ids()
    return players


import os
for cnt,filename in enumerate(os.listdir(r'D:/Users/emill/shitter/a')):
    with open('parsed_names.csv','a',newline='\n')as f:
        thewriter = csv.writer(f)
        thewriter.writerow([filename])
    if filename.endswith(".json"):
        steamid = 234
        print(filename,round(cnt/len(os.listdir(r'D:\Users\emill\shitter/a')),0))
        players_this_game = get_names(filename)
        just_name = [x[0] for x in players_this_game]

        if filename.endswith('.json'):
            json_name = filename
            df = pd.read_csv("log (3).csv")

            for i in df["name"]:
                if i in just_name:
                    steamid = [x[1] for x in players_this_game if i == x[0]][0]
                    main(json_name, steamid)


# Clean games
"""
import os
for cnt,filename in enumerate(os.listdir(r'C:/Users\emill\PycharmProjects\csgoparse\csgo\examples/bsd')):
    steamid = 76561198055893769
    if filename.endswith('.json'):
        json_name = filename
        try:
            main(json_name, steamid)
        except Exception as e:
            print(e)"""