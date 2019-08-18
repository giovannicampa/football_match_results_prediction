# Function that reads the csv files of the matches and returns a dataframe with the features and the target
def prepare_data():
    
    import pandas as pd
    import os
    import numpy as np
    import pickle

    # - Reading all the files in the folder and generating the prepared dataframes
    
    # Initialisation of the dataframe for all the data (features + target)
    data_all_seasons = pd.DataFrame()

    # List of all csv files with the information of the matches
    csv_list = os.listdir('archive')



    # - Calculating the features for each csv file and putting it into a dataframe
    
    # Iterating over all the .csv files in the folder. At the end of the loop they will be put together
    for file in range(len(csv_list)):
    
        # File name
        target = ('archive/' + csv_list[file])
        
        season = pd.read_csv(target)
        season.Date.astype('category')
        
        giornate = pd.unique(season.Date)
        
        # Renaming the dates to game dates (instead of 01/08/17, we will have day 0,1,2...)
        for giornata in range(len(giornate)):
            season.Date[season.Date == giornate[giornata]] = giornata
        
        # Finding all team namesand sorting alphabetically
        teams = pd.unique(season.HomeTeam)
        teams.sort()
        
        # Initialisation: Table with rows: progressive game days, columns: team names, content: points
        points_table = np.zeros([len(giornate),len(teams)])
        
        # Initialisation: Table with rows: progressive game days, columns: team names, content: ranking
        ranking_table = np.zeros([len(giornate),len(teams)])
        
        
        # Two tables columns = teams, rows = game days (filled row wise)
        for giornata in range(len(giornate)):
            
            # First table: nr of points by team every day
            for team in range(len(teams)):
                isHome = season.HomeTeam == teams[team]
                isAway = season.AwayTeam == teams[team]
                isDay = season.Date <= giornata
                
                punti_asHome = np.sum(season.loc[(isHome & isDay), 'FTHG'])
                punti_asAway = np.sum(season.loc[(isAway & isDay), 'FTAG'])
                
                points_table[giornata, team] = max(punti_asHome, punti_asAway)
            
            points_today = points_table[giornata, :]
            points_today = sorted(np.unique(points_today), reverse = True)
                
            
            # Second table: ranking of the team by points day by day (calculated using the first table)
            for point_class in range(len(points_today)):
                
                nr_teams_same_points = sum(points_table[giornata, :] == points_today[point_class])
                
                ranking_table[giornata, points_table[giornata,:] == points_today[point_class]] = point_class
                
        
        # Storing the two tables in dataframes for easier access        
        points_df = pd.DataFrame(points_table)
        points_df.columns = teams
        
        ranking_df = pd.DataFrame(ranking_table)
        ranking_df.columns = teams
        
        # The ranking that we calculated is the ranking at the end of the day, so we need to shift it of one day, to get
        # the teams' ranking on the day in which they played
        
        ranking_df.shift(periods = 1)
        
        
        # - Creating the table that will be used for the prediciton model
        
        nr_games = season.shape[0]
        
        table_names = ['rank','mean_points_per_game_tot', 'mean_points_per_game_in_location', 'games_won_tot','mean_shots','mean_SOT','mean_fouls','mean_yellow_cards','score']
        locations = ['_home','_away']
        
        table_names_all = []
        for name in table_names:
            for location in locations:
                table_names_all.append(name+location)
              
        
        season_table = np.zeros([nr_games, len(table_names_all)]) # 1st 2 is the nr of teams, 2nd 2 is the target features (2 scores)
        
        season_data = pd.DataFrame(season_table)
        season_data.columns = table_names_all
        
        
        # Iterating over all the played games in the dataset
        for game in range(nr_games):
            
            # This the progressive game number and is kept equal through all the dataframes
            game_day = season.loc[game, 'Date']
            
            team_home = season.loc[game, 'HomeTeam']
            team_away = season.loc[game, 'AwayTeam']
        
        
            # - Feature: team rank
        
            season_data.rank_home[game] = ranking_df.loc[game_day, team_home]
            season_data.rank_away[game] = ranking_df.loc[game_day, team_away]
        
            
            # - Feature: mean scores in the same location
        
            # Indices to find the various elements for which I want to calculate the mean
            idx_Date = season.Date <= game_day
        
            idx_Home_at_Home = season.HomeTeam  == team_home
            idx_Away_at_Away = season.AwayTeam  == team_away
        
            idx_Home_at_Away = season.AwayTeam  == team_home
            idx_Away_at_Home = season.HomeTeam  == team_away
            
            # Points in same situation
            points_home_at_home = season.loc[(idx_Date & idx_Home_at_Home), 'FTHG'].values.tolist()
            points_away_at_away = season.loc[(idx_Date & idx_Away_at_Away), 'FTAG'].values.tolist()
            
            # Mean points of each team playing in the same situation
            mean_pts_home_as_home = np.mean(points_home_at_home)
            mean_pts_away_as_away = np.mean(points_away_at_away)
        
            # Assigning to dataframe
            season_data.mean_points_per_game_in_location_home[game] = mean_pts_home_as_home
            season_data.mean_points_per_game_in_location_away[game] = mean_pts_away_as_away
        
        
            # - Feature: mean scores in all locations (total)
            
            # Points in the opposite situation
            points_home_at_away = season.loc[(idx_Date & idx_Home_at_Away), 'FTAG'].values.tolist()
            points_away_at_home = season.loc[(idx_Date & idx_Away_at_Home), 'FTHG'].values.tolist()
            
            # Mean points on all the games for the home team (same and opposite situation)
            mean_pts_home_tot = np.nanmean(points_home_at_away + points_home_at_home)        
                
            # Mean points on all the games for the away team (same and opposite situation
            mean_pts_away_tot = np.nanmean(points_away_at_home + points_away_at_away)
            
            season_data.mean_points_per_game_tot_home[game] = mean_pts_home_tot
            season_data.mean_points_per_game_tot_away[game] = mean_pts_away_tot
        
        
            # - Feature: games_won
            
            season_data.games_won_tot_home[game] = np.mean((season.loc[(idx_Date & idx_Home_at_Home), 'FTR'].values == 'H').tolist() + (season.loc[(idx_Date & idx_Home_at_Away), 'FTR'].values == 'A').tolist())
            season_data.games_won_tot_away[game] = np.mean((season.loc[(idx_Date & idx_Away_at_Away), 'FTR'].values == 'A').tolist() + (season.loc[(idx_Date & idx_Away_at_Home), 'FTR'].values == 'H').tolist())
            
            # - Feature: mean shots (all locations)
            
            season_data.mean_shots_home[game] = np.mean(season.loc[(idx_Date & idx_Home_at_Home), 'HS'].values.tolist() + season.loc[(idx_Date & idx_Home_at_Away), 'AS'].values.tolist())
            season_data.mean_shots_away[game] = np.mean(season.loc[(idx_Date & idx_Away_at_Away), 'AS'].values.tolist() + season.loc[(idx_Date & idx_Away_at_Home), 'HS'].values.tolist())
            
            
            # - Feature: mean shot on target (all locations)
            
            season_data.mean_SOT_home[game] = np.mean(season.loc[(idx_Date & idx_Home_at_Home), 'HST'].values.tolist() + season.loc[(idx_Date & idx_Home_at_Away), 'AST'].values.tolist())
            season_data.mean_SOT_away[game] = np.mean(season.loc[(idx_Date & idx_Away_at_Away), 'AST'].values.tolist() + season.loc[(idx_Date & idx_Away_at_Home), 'HST'].values.tolist())
            
            
            # - Feature: mean fouls (all locations)
            
            season_data.mean_fouls_home[game] = np.mean(season.loc[(idx_Date & idx_Home_at_Home), 'HF'].values.tolist() + season.loc[(idx_Date & idx_Home_at_Away), 'AF'].values.tolist())
            season_data.mean_fouls_away[game] = np.mean(season.loc[(idx_Date & idx_Away_at_Away), 'AF'].values.tolist() + season.loc[(idx_Date & idx_Away_at_Home), 'HF'].values.tolist())
            
            
#            # - Feature: mean corners (all locations)
#            
#            season_data.mean_corners_home[game] = np.mean(season.loc[(idx_Date & idx_Home_at_Home), 'HC'].values.tolist() + season.loc[(idx_Date & idx_Home_at_Away), 'AC'].values.tolist())
#            season_data.mean_corners_away[game] = np.mean(season.loc[(idx_Date & idx_Away_at_Away), 'AC'].values.tolist() + season.loc[(idx_Date & idx_Away_at_Home), 'HC'].values.tolist())
            
            
            # - Features: mean yellow cards (all locations)
            
            season_data.mean_yellow_cards_home[game] = np.mean(season.loc[(idx_Date & idx_Home_at_Home), 'HY'].values.tolist() + season.loc[(idx_Date & idx_Home_at_Away), 'AY'].values.tolist())
            season_data.mean_yellow_cards_away[game] = np.mean(season.loc[(idx_Date & idx_Away_at_Away), 'AY'].values.tolist() + season.loc[(idx_Date & idx_Away_at_Home), 'HY'].values.tolist())
            
            
            # - Assigning the target values
            season_data.score_home[game] = season.FTHG[game]
            season_data.score_away[game] = season.FTAG[game]
        
        
        # - Putting the dataframes for the prediction model together
        data_temp = season_data.iloc[5:]
        
        if file == 0:
            data_all_seasons = data_temp
        else:
            data_all_seasons = pd.concat([data_all_seasons, data_temp])
        
    
    # - Removing NaNs
    data_all_seasons.dropna(axis = 0, inplace = True)
    
    
    # Adding a column of the winners for each team
    winners = np.empty([data_all_seasons.shape[0],1], dtype = 'str')
    winners[data_all_seasons.score_home > data_all_seasons.score_away] = 'H'
    winners[data_all_seasons.score_home < data_all_seasons.score_away] = 'A'
    winners[data_all_seasons.score_home == data_all_seasons.score_away] = 'D'
    
    data_all_seasons['winners'] = winners
    
    
    # - Saving the dataframe
    pickle_out = open("data_all_seasons.pickle", "wb")
    pickle.dump(data_all_seasons, pickle_out)
    pickle_out.close()
        
    # - Data will also be returned as output of the function
    return season_data
    
    