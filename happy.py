import json
import pandas as pd
import csv
import os

class Killparser():
    """
    PARSES TICKS LEADING UP TO A KILL AND SPLITS THEM INTO CSVs
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
        for t in range(2):
            if t == 0:
                team_index = "T"
            else:
                team_index = "CT"
            for i in range(5):
                players.append((self.x["GameRounds"][1]["Frames"][0][team_index]["Players"][i]["Name"],self.x["GameRounds"][1]["Frames"][1][team_index]["Players"][i]["SteamId"]))
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


        for rounds in range(self.n_rounds):
            for frames in range(len(self.x["GameRounds"][rounds]["Frames"])):
                for team in range(2):
                    for player in range(4):
                        if team == 1:
                            team_index = "T"
                        else:
                            team_index = "CT"
                        try:
                            if str(self.x["GameRounds"][rounds]["Frames"][frames][team_index]["Players"][player]["SteamId"]) == steamid:

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
                    df.to_csv(f"{kill_folder}{self.MatchId}.csv", mode='a', header=False)
            except Exception as e:
                print(e)

    def split_df_by_kill(self,mainfile,killfile,steamid,out_folder,write_to_csv=False):
        """
        :param attacker_name:
        :return: List of dfs consisting of ticks up to the kill
        """
        retruning_list = []

        df1 = pd.read_csv(mainfile)
        df2 = pd.read_csv(killfile)
        df2 = df2[df2["AttackerSteamId"] == int(steamid)]
        # drop dupes
        df2 = df2.drop_duplicates()
        # Check how many files in output dir so to know what name to give file
        n_files = len([name for name in os.listdir(out_folder)])
        print("n_files",n_files)

        for cnt, x in enumerate(range(len(df2))):
            # Get the next kill and append to list
            mintick = min(df2["TICK"])
            out_df = df1[df1['TICK'] <= mintick]

            if len(out_df)>0:
                retruning_list.append(out_df)
                if write_to_csv is True:
                    out_df.to_csv(f"{out_folder}/{n_files + cnt}.csv")

            # Cut off kill from big dfs
            df1 = df1[df1['TICK'] > mintick]
            df2 = df2[df2['TICK'] > mintick]

        return retruning_list


def main(json_input_folder,json_name, steamid, out_folder):
    """
    Takes in a parsed json file and a player name. Creates CSV files from each kill, having
    the ticks before the kill happened.

    :param json_name:
    :param steamid:
    :return:
    """
    datamover = Killparser()
    json_name_without_end = json_name.replace(".json", "")
    mainfile = f"{out_folder}/games/{json_name_without_end}.csv"
    killfile = f"{out_folder}/kills/{json_name_without_end}.csv"

    datamover.read_json(f"{json_input_folder}/{json_name}")
    datamover.get_pos_player(steamid, mainfile)
    datamover.get_kills_csv(killfile)


    # mainfile = f'D:/Users/emill/csgocheaters/games/{json_name_without_end}.csv'
    # killfile = f'D:/Users/emill/csgocheaters/kills/{json_name_without_end}.csv'

    datamover.split_df_by_kill(mainfile, killfile, steamid, out_folder, write_to_csv=True)


def get_names(json_name):
    datamover = Killparser()
    datamover.read_json(f"C:/Users/emill/PycharmProjects/csgoparse/csgo/examples/bsd/{json_name}")
    players = datamover.get_players_ids()
    return players


def csv_steamid(json_input_folder, output_folder):
    """
    json_folder: Folder with parsed demos (json files)
    output_folder: Folder where the outputs are made
    """

    for cnt,filename in enumerate(os.listdir(json_input_folder)):
        steamid = 42
        if filename.endswith('.json'):
            json_name = filename

            df = pd.read_csv("CHEATERURLS2.csv")
            print("JSONNAME",json_name)
            for i in range(len(df["game"])):
                x = df["game"].iloc[i]
                game = x.split('/')[2]
                game = game.split('.')[0]
                if json_name == f"{game}.json":
                    cheaters_id = df["profile"].iloc[i]
                    print(cheaters_id)

                    if cheaters_id[-1] != "/":
                        id = cheaters_id.split("/")[4]
                        print("id",id)

                        steamid = id
                        try:
                            main(json_input_folder,json_name, steamid)
                        except Exception as e:
                            print(e)




def json_to_kill_csv(json_input_folder,output_folder,steamid):
    """
    json_input_folder: Folder with parsed demos (json files)
    output_folder: Folder where the outputs are created
    """
    import os
    for cnt, filename in enumerate(os.listdir(json_input_folder)):
        if filename.endswith('.json'):
            json_name = filename
            try:
                main(json_input_folder, json_name, steamid, output_folder)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    json_input_folder = 'C:/Users/emill/PycharmProjects/csgoparse/csgo/examples/bsd'
    out_folder = "D:/Users/emill/csgocheaters/cleankills/"
    steamid = 76561198134270402
    json_to_kill_csv(json_input_folder,out_folder,steamid)