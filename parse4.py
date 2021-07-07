import json
import pandas as pd
import csv
import numpy as np
from math import pi


class Killparser():
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
                            #str(self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]["SteamId"]) == steamid:

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
        retruning_list = []

        enormous = []

        df1 = pd.read_csv(mainfile)
        df2 = pd.read_csv(killfile)

        df3 = df2[df2["AttackerSteamId"] == int(steamid)]
        victims = [x for x in df3["VictimSteamId"]]
        # Remove duplicates
        victims = set(victims)
        victims = list(victims)
        for p in victims:
            # Read main file with all ticks
            df = pd.read_csv(mainfile)
            print(df[df["SteamId"] == int(steamid)], "==", steamid)

            df1 = df[df["SteamId"] == int(steamid)]
            # Only the victims frames
            print(df[df["SteamId"]== p],"==",p)
            df2 = df[df["SteamId"] == p]

            #df1 = df1.set_index("TICK")

            df2 = df2[["TICK", "X", "Y", "Z"]]
            #df2 = df2.set_index("TICK")

            print(df2)





            # Convert 0-360 range to -180 - 180 range for easier trig
            df1['ViewX'] = (df1['ViewX'] + 180) % 360 - 180
            df1['ViewY'] = (df1['ViewY'] + 180) % 360 - 180



            big = df1.merge(df2, on='TICK')
            enormous.append(big)

        enormous[1].to_csv("first.csv")
        enormous[2].to_csv("second.csv")
        for cnt,i in enumerate(enormous):
            if cnt != 0:
                enormous[0].append(i)

        print("CHONKY ALERT")
        print(enormous[0])
        print("CHONKY ALERT")


        print(enormous)

        big = enormous[0]

        x1 = big["X_x"]
        y1 = big["Y_x"]
        z1 = big["Z_x"]

        x2 = big["X_y"]
        y2 = big["Y_y"]
        z2 = big["Z_y"]

        xdif = np.array(abs(x2 - x1))
        ydif = np.array(abs(y2 - y1))
        zdif = np.array(abs(z2 - z1))





        rightLeft = np.arctan2(ydif, xdif) * (180 / pi)
        hypot3D = np.sqrt(zdif ** 2 + xdif ** 2 + ydif ** 2)
        upDown = (-np.arcsin(zdif / hypot3D)) * (180 / pi)




        big["X_Off_By_Degrees"] = rightLeft - big["ViewX"]
        big["Y_Off_By_Degrees"] = upDown - big["ViewY"]
        big.to_csv("nextpls.csv")

        #df2 = df2[df2["AttackerSteamId"] == int(steamid)]

        # drop dupes

        #df2 = df2.drop_duplicates()
        n_files = len([name for name in os.listdir(r'D:\Users\emill\csgocheaters\singlekills/')])
        print("n_files",n_files)

        df1 = big
        df2 = pd.read_csv(killfile)

        for cnt, x in enumerate(range(len(df2))):
            # Get the next kill and append to list
            mintick = min(df2["TICK"])
            out_df = df1[df1['TICK'] <= mintick]

            if len(out_df)>0:
                retruning_list.append(out_df)
                if write_to_csv == True:
                    out_df.to_csv(f"{out_folder}/{n_files + cnt}.csv")

            # Cut off kill from big dfs
            df1 = df1[df1['TICK'] > mintick]
            df2 = df2[df2['TICK'] > mintick]

        return retruning_list


def main(json_name, steamid):
    """
    Takes in a parsed json file and a player name. Creates CSV files from each kill, having
    the ticks before the kill happened.
    :param json_name:
    :param steamid:
    :return:
    """
    datamover = Killparser()
    datamover.read_json(f"C:/Users/emill/PycharmProjects/csgoparse/128dirty/{json_name}")
    datamover.get_pos_player(f"{steamid}",f"D:/Users/emill/csgocheaters/games/")
    datamover.get_kills_csv(r'D:\Users\emill\csgocheaters\kills/')

    json_name_without_end = json_name.replace(".json","")
    mainfile = f'D:/Users/emill/csgocheaters/games/{json_name_without_end}.csv'
    killfile = f'D:/Users/emill/csgocheaters/kills/{json_name_without_end}.csv'
    out_folder = r"D:\Users\emill\csgocheaters\singlekills/"
    datamover.split_df_by_kill(mainfile, killfile, steamid, out_folder, write_to_csv=True)


def get_names(json_name):
    datamover = Killparser()
    datamover.read_json(f"C:/Users/emill/PycharmProjects/csgoparse/128dirty/{json_name}")
    players = datamover.get_players_ids()
    return players


import os
for cnt,filename in enumerate(os.listdir(r'C:\Users\emill\PycharmProjects\csgoparse\128dirty/')):
    if cnt >4:
        steamid = 234
        print(filename,round(cnt/len(os.listdir(r'C:\Users\emill\PycharmProjects\csgoparse\128dirty/')),0))
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