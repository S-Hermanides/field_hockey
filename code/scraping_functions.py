from bs4 import BeautifulSoup
import numpy as np
from selenium.webdriver.support.ui import Select
import time


def get_data(driver):
    """Extract a dictionary with Team Name as key and Stat Value as value from the selected page"""
    soup = BeautifulSoup(driver.page_source, 'lxml')
    data_table = soup.find('table', {'id': 'rankings_table'})
    rows = [row for row in data_table.findAll('tr')]
    rows = rows[1:]  # Table has a header row, which is excluded here
    stat_dict = {}
    for row in rows:
        row_data = [entry.text.strip() for entry in row.findAll('td')]
        if len(row_data) > 1:
            stat_dict[row_data[1].split('(')[0].strip()] = row_data[-1]
    return stat_dict


def add_to_result(data_dict, result_dict):
    for team in result_dict:
        if team in data_dict:
            result_dict[team].append(data_dict[team])
        else:
            result_dict[team].append(np.nan)


def get_stats(driver, result_dict, header_list, stats_list, teams_list):
    """
    Loops through the stats for the 2019 season and creates a dictionary with teams as keys
     and a list of stats as values
     """
    for stat in header_list[2:]:
        if stat in stats_list:
            dropdown_stats2 = Select(driver.find_element_by_id('stat_seq'))  # Different dropdown sequence after new selection
            dropdown_stats2.select_by_visible_text(stat)
            time.sleep(1)
            dropdown_display = Select(driver.find_element_by_name('rankings_table_length'))
            dropdown_display.select_by_value('-1')
            add_to_result(get_data(driver), result_dict)
        else:
            for team in teams_list:
                result_dict[team].append(np.nan)
    return result_dict


def get_games(driver):
    """Extract the number of games per season from the data table"""
    soup = BeautifulSoup(driver.page_source, 'lxml')
    data_table = soup.find('table', {'id': 'rankings_table'})
    rows = [row for row in data_table.findAll('tr')]
    rows = rows[1:]  # Table has a header row, which is excluded here
    games_dict = {}
    for row in rows:
        row_data = [entry.text.strip() for entry in row.findAll('td')]
        if len(row_data) > 1:
            games_dict[row_data[1].split('(')[0].strip()] = row_data[2]
    return games_dict

