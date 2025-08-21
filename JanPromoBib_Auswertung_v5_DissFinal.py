#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
JanPromoBib_Auswertung_v5.py 

Author: Jan Heysel
Version: v5 (Stand: 10.10.24)

Diese Bibliothek enthält Funktionen, die zur Auswertung im Rahmen der Promotion von Jan Heysel implementiert und genutzt wurden. 
"""

# Bibliotheken
import numpy as np 
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

from decimal import Decimal
import sys
import logging # see https://stackoverflow.com/questions/49580313/create-a-log-file
from datetime import datetime

# zusätzlich aus Schritt 1:
import codecs
import pylab
import json
import re
#from tqdm import tqdm

from collections import Counter

# zum Plotten der Sanky Diagramme:
from pysankey2.datasets import load_fruits
from pysankey2 import Sankey


#####################################################################################
##################### Funktionen für die überarbeitere Analyse ######################
#####################       beim Finalisieren der Diss        #######################
#####################################################################################

def make_df_Sankey_from_df_all(col1, col2, df_all):
    df_Sankey = df_all[[col1, col2]].astype(int)
    df_Sankey = df_Sankey[(df_Sankey[col1] != (4 or NaN)) & (df_Sankey[col2] != (4 or NaN))]
    N_filtered = df_Sankey.shape[0]    
    df_Sankey.rename(columns={col1: 'layer1'}, inplace=True)
    df_Sankey.rename(columns={col2: 'layer2'}, inplace=True)
    df_Sankey.replace(0, "N0", inplace=True)
    df_Sankey.replace(1, "N1", inplace=True)
    df_Sankey.replace(2, "N2", inplace=True)
    df_Sankey.replace(3, "N3", inplace=True)
    return df_Sankey, N_filtered

# es scheint, als müsste ich die Anteile selber berechnen:
def calc_freq_Sankey(df_Sankey):

    N_filtered = df_Sankey.shape[0]

    N_pre_N0 = len(df_Sankey[df_Sankey['layer1'].isin(["N0"])])
    N_pre_N1 = len(df_Sankey[df_Sankey['layer1'].isin(["N1"])])
    N_pre_N2 = len(df_Sankey[df_Sankey['layer1'].isin(["N2"])])
    N_pre_N3 = len(df_Sankey[df_Sankey['layer1'].isin(["N3"])])

    N_post_N0 = len(df_Sankey[df_Sankey['layer2'].isin(["N0"])])
    N_post_N1 = len(df_Sankey[df_Sankey['layer2'].isin(["N1"])])
    N_post_N2 = len(df_Sankey[df_Sankey['layer2'].isin(["N2"])])
    N_post_N3 = len(df_Sankey[df_Sankey['layer2'].isin(["N3"])])

    row_Sankey = np.array([N_pre_N0, N_pre_N1, N_pre_N2, N_pre_N3, N_post_N0, N_post_N1, N_post_N2, N_post_N3])
    row_Sankey = row_Sankey / N_filtered * 100
    row_Sankey = np.round(row_Sankey, decimals=1)
    row_Sankey = np.append(row_Sankey, N_filtered)

    return row_Sankey




def Daten_aus_df_all_holen(code, df_all):
    
    pre_label  = code + "Niv_pre"
    post_label = code + "Niv_post"
    
    df = df_all[[pre_label, post_label]].astype(int)
    df = df[(df[pre_label] != (4 or NaN)) & (df[post_label] != (4 or NaN))]
    df = df.rename(columns={pre_label: 'pre', post_label: 'post'})
    N_filtered = df.shape[0]  
    
    # DataFrame ins Long-Format umwandeln
    df_long = pd.melt(df, var_name='Testzeitpunkt', value_name='Niveau')
    df_long['Testzeitpunkt'] = df_long['Testzeitpunkt'].replace({'pre': 1, 'post': 2}).astype(int)
        
    pre = df["pre"]
    post = df["post"]

    return df, df_long, pre, post, N_filtered


def ClusterDaten_aus_df_all_holen(code, df_all):
    
    pre_label  = code + "_median_pre"
    post_label = code + "_median_post"
    
    df = df_all[[pre_label, post_label]].astype(int)
    df = df[(df[pre_label] != (4 or NaN)) & (df[post_label] != (4 or NaN))]
    df = df.rename(columns={pre_label: 'pre', post_label: 'post'})
    N_filtered = df.shape[0]  
    
    # DataFrame ins Long-Format umwandeln
    df_long = pd.melt(df, var_name='Testzeitpunkt', value_name='Niveau')
    df_long['Testzeitpunkt'] = df_long['Testzeitpunkt'].replace({'pre': 1, 'post': 2}).astype(int)
        
    pre = df["pre"]
    post = df["post"]

    return df, df_long, pre, post, N_filtered



def Korrelationsplot_zeichnen_speichern(code, df_long, Ordner_Ziel_Abbildungen):

    sns.set(style="darkgrid")

    # Häufigkeiten der Datenpunkte berechnen
    data_points = list(zip(df_long['Testzeitpunkt'], df_long['Niveau']))
    counts = Counter(data_points)
    
    # Extrahiere die einzigartigen (Testzeitpunkt, Niveau) Werte und deren Häufigkeiten
    unique_data = pd.DataFrame(list(counts.keys()), columns=['Testzeitpunkt', 'Niveau'])
    unique_data['Häufigkeit'] = list(counts.values())
    
    # Farben definieren basierend auf dem Niveau-Wert
    color_map = {
        0: '#006BA4',  # Farbe für Niveau 0
        1: '#FF800E',  # Farbe für Niveau 1
        2: '#ABABAB',  # Farbe für Niveau 2
        3: '#595959'   # Farbe für Niveau 3
    }
    
    # Farben für die einzelnen Punkte zuordnen
    colors = [color_map[niveau] for niveau in unique_data['Niveau']]
    
    
    Schriftgroesse = 40
    
    # Streudiagramm erstellen
    plt.figure(figsize=(10, 10))
    plt.scatter(unique_data['Testzeitpunkt'], unique_data['Niveau'],  
                s=unique_data['Häufigkeit'] * 100,  # Punktgröße proportional zur Häufigkeit (dok: s="The marker size in points**2 (typographic points are 1/72 in.)"
                color=colors, alpha=0.6, edgecolors='w')
    
    
    
    # Anpassungen der Achsen:
    plt.xlim(0.5, 2.5)  # Bereich der y-Achse auf 0.5 bis 2.5 setzen
    plt.xticks([1, 2], ['Pre (1)', 'Post (2)'], fontsize=Schriftgroesse)  # Werte 1 und 2 mit Labels versehen
    
    plt.ylim(-0.2, 3.2)  # Bereich der y-Achse auf 0.5 bis 2.5 setzen
    plt.yticks([0, 1, 2, 3], ['N0 (0)', 'N1 (1)', 'N2 (2)', 'N3 (3)'], fontsize=Schriftgroesse)  # Werte 1 und 2 mit Labels versehen
    
    
    # Titel und Achsenbeschriftungen hinzufügen
    #plt.title('Testzeitpunkt gegen Niveaus (Punktgröße proportional zur Häufigkeit)')
    plt.xlabel('Testzeitpunkt', fontsize=Schriftgroesse)
    plt.ylabel('Niveau', fontsize=Schriftgroesse)
    plt.grid(True)

    # Reduce the spacing around the figure
    plt.tight_layout()  # Automatically adjust layout for minimal spacing
    # Alternatively, you can fine-tune with subplots_adjust if needed
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust these values as needed

    
    plt.savefig(Ordner_Ziel_Abbildungen + "/" + code + "_Korrelationsplot.pdf", format='pdf')
    plt.show()

    print("Korrelationsplot zu " + code + " gespeichert.")
    return 1

    
def berechne_z_Jan(df):

    # all values of the df:
    unique_values = np.unique(df.values)
    
    # maximum niveau:
    Nmax = np.max(unique_values)
    
    # maximum and real increase ("Zunahme") of the niveaus of each person:
    df["Z_max"]  = Nmax - df["pre"]
    df["Z_real"] = df["post"] - df["pre"]
    
    # sums over all persons:
    sum_max = df['Z_max'].sum()
    sum_real = df['Z_real'].sum()
    
    # final parameter:
    Z = sum_real / sum_max

    return Z



def Histogram_Geschlechtervergleich_erstellen(m, w, code, Ordner_Ziel_Abbildungen):

    # Bins und deren Ränder passend zum minimalen Wert der Niveauverschiebung setzten:
    min = np.min(np.concatenate((w, m)))
    #print(min)
    binedges = np.arange(min-0.5, 3.5+1)
    bins = int(4+np.abs(min))
    x_Mitte = np.arange(min, 3+1)
    delta = 0.2
    delta_array = np.full(bins, delta)
    x_rechts = x_Mitte + delta_array
    x_links  = x_Mitte - delta_array
    #print(binedges, bins, x_Mitte, delta_array, x_rechts, x_links)

    
    # Werte des Histogramms berechnen
    N_Maedels = len(w)
    N_Jungs = len(m)
    m_hist_rel = np.histogram(m, bins=binedges)[0] / N_Jungs * 100
    w_hist_rel = np.histogram(w, bins=binedges)[0] / N_Maedels * 100
    
    # Farben setzen:
    sns.set_theme()  # <-- This actually changes the look of plots.

    rot1='#C1392B'
    rot2='#D75C4A'
    blau1='#005582'
    blau2='#005082'

    # Plotten:
    plt.bar(x = x_rechts, height = m_hist_rel, color= blau1, alpha=0.6, label="männlich", width=delta*2) 
    plt.bar(x = x_links, height = w_hist_rel, color= rot1, alpha=0.6, label="weiblich", width=delta*2) 
    
    # Werte über den männlichen Säulen anzeigen
    for i in range(len(m_hist_rel)):
        plt.text(x_rechts[i], m_hist_rel[i] + 0.01, f'{m_hist_rel[i]:.1f}', ha='center', va='bottom')
    
    # Werte über den weiblichen Säulen anzeigen
    for i in range(len(w_hist_rel)):
        plt.text(x_links[i], w_hist_rel[i] + 0.01, f'{w_hist_rel[i]:.1f}', ha='center', va='bottom')
    
    # Add labels and title
    plt.xticks(x_Mitte)
    plt.xlabel(r'Verteilung von $\Delta \tilde{\mathcal{N}}_{\mathcal{C}}$ bei Jungen und Mädchen')#Lernzuwachs im Geschlechtervergleich') # # Differenzen der Cluster-Median-Niveaus $\Delta \mathcal{N}_{Median}$
    plt.ylabel('relative Häufigkeit in Prozent')
    #plt.title('Geschlechtervergleich in Cluster ' + code)
    plt.title('Lernzuwachs im Geschlechtervergleich, Cluster ' + code)
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.savefig(Ordner_Ziel_Abbildungen + "Geschlechtervergleich_" + code + "_hist.pdf")
    plt.show()

    return w_hist_rel, m_hist_rel
    
def bootstrap_array(m, n=None):
    """
    Erzeugt ein Bootstrap-Array aus dem gegebenen Array m.
    
    Parameter:
    - m: Das ursprüngliche Array (1D-Array)
    - n: Anzahl der Ziehungen für das Bootstrap-Array (Standard ist die Länge von m)
    
    Rückgabe:
    - Ein Bootstrap-Array mit den gleichen Werten wie m, jedoch zufällig gezogen.
    """
    # Wenn n nicht angegeben ist, setze es auf die Länge von m
    if n is None:
        n = len(m)
    
    # Erzeuge das Bootstrap-Array durch zufälliges Ziehen mit Zurücklegen
    bootstrap_sample = np.random.choice(m, size=n, replace=True)
    
    return bootstrap_sample


def power_mannwhitney_bootstrap(m,w, alpha, num_simulations):
   
    # Bestimme die Längen der Eingangs-Arrays:
    n_m = len(m)
    n_w = len(w)
    
    # Zähler für signifikante Ergebnisse
    significant_results = 0
    
    for _ in range(num_simulations):
        # Simuliere Daten für X und Y
        X = bootstrap_array(w)
        Y = bootstrap_array(m)
        
        # Führe den Mann-Whitney-U-Test durch
        stat, p_value = mannwhitneyu(X, Y, alternative='two-sided')
        
        # Überprüfe, ob das Ergebnis signifikant ist
        if p_value < alpha: # in diesen Fällen handelt es sich um eine falsche Annahme von H0
            significant_results += 1
    
    # Berechne den beta-Fehler unter der Annahme, dass es eigentlich keinen Effekt gibt
    beta = significant_results / num_simulations # der Anteil falscher Annahmen von H0 = die Wahrscheinlichkeit für eine falsche Annahme von H0 = beta
    power = 1 - beta
    return power


#####################################################################################
##################### Funktionen für Schritt 2 der Auswertung #######################
#####################    Zuordnung: Kategorien --> Niveaus    #######################
#####################################################################################

def extract_number(s):
    # Search for a pattern that matches a digit using regular expression
    match = re.search(r'\d', s)
    
    if match:
        # Extract and return the matched digit
        return int(match.group())
    else:
        # Return a default value (or raise an error) if no digit is found
        return None


### allgemein:

def read_prepare_Kreuztabelle(Ordner, dateiname, prepost):
    """
    Funktion, die eine Datendatei als Pandas Dataframe einliest und transponiert
    """
    # read file:
    df = pd.read_excel(Ordner + "/" + dateiname)

    # transpose, rename columns, remove unnecessary rows:
    df = df.rename(columns={'Unnamed: 0': 'P'})
    df = df.transpose()
    df.columns = df.iloc[0].astype(str)
    df = df.drop(["P"])
    df = df.drop(["Total"])

    # Index aufbereiten:
    df.index = df.index.str.strip() # removes whitespaces on both sides
    #df.index = df.index.str.rstrip(prepost) # removes "pre" or "post" on the right side
    df.index = df.index.str.rstrip("pre")
    df.index = df.index.str.rstrip("post")
    df.index = df.index.str.lstrip("P") # removes "P" on the left side
    df.index = df.index.astype(int) # casts to int

    # sort dataframe by index:
    df = df.sort_index()  
        
    return df


def zaehle_Anz_Kat_je_Niveau(df, code):
    """
    Takes a dataframe df and a code from the list "codes" like "M1". 
    Reads the column "Niveau" from the sheet "code" from "Codesystem_alle_mitKreuztabellenmerge.xlsx".
    Counts how often a category from each niveau appears in each answer. 
    Notes this down in the new columns "Anz_N0", "Anz_N1", "Anz_N2", "Anz_N3". 
    Returns this dataframe.
    """
    
    df_Codesystem = pd.read_excel("Codesystem_alle_mitKreuztabellenmerge.xlsx", sheet_name=code)
    
    # Bilde arrays mit den Kategorien, die zu den einzelnen Niveaus gehören:
    df_Codesystem_N3 = df_Codesystem[df_Codesystem['Niveau'] == 3]
    codes_N3 = df_Codesystem_N3["Kürzel"].values

    df_Codesystem_N2 = df_Codesystem[df_Codesystem['Niveau'] == 2]
    codes_N2 = df_Codesystem_N2["Kürzel"].values

    df_Codesystem_N1 = df_Codesystem[df_Codesystem['Niveau'] == 1]
    codes_N1 = df_Codesystem_N1["Kürzel"].values

    df_Codesystem_N0 = df_Codesystem[df_Codesystem['Niveau'] == 0]
    codes_N0 = df_Codesystem_N0["Kürzel"].values
    
    # Nun im Dataframe df je Spalten hinzufügen (Anz_Kat_N0 bis Anz_Kat_N3), wie viele Kategorien eines Niveaus eine Antwort enthielt. 
    # addiere dazu alle Spalteneinträge aus den Spalten je eines Arrays auf:

    # Nun Spalten für die Niveauzuordnungen hinzufügen:
    liste_niveau_codes = [codes_N0, codes_N1, codes_N2, codes_N3]
    liste_neue_spalten = ["Anz_N0", "Anz_N1", "Anz_N2", "Anz_N3"]

    for i in range(4):
        # erst ein Teil-df bilden, das nur die Spalten enthält, die in codes_N3 bezeichnet sind. Dann die Summe über das Teil-df im Gesamt-df als neue Spalte hinzufügen:
        df[liste_neue_spalten[i]] = df[liste_niveau_codes[i]].sum(axis=1).astype(int)
        # df1["Anz_N3"].sum(axis=0) # Anzahl, wie oft eine Kategorie dieses Niveaus gesetzt wurde
        
    return df


#####################################################################################
#####################     Katgorien zu Niveaus zuordenen      #######################
#####################################################################################

def niveaus_aus_kategorien(df, code, N3_Schwellen):
    """
    Diese Funktion ordnet für alle Kategoriensysteme M1 bis PR3 allen Personen (=Zeilen im df) aufgrund ihrer inhaltlichen Kategorien ein Niveau zu.
    
    df: pandas Dataframe, in dem die Personen die Zeilen sind und die Spalten die inhaltichen Kategorien umfassen. Jede Kategorienbezeichnung enthält auch ein zugeordnetes Niveau.
    code: der Code M1 bis PR3
    N3_Schwellen: das dict N3_Schwellen = {"S1":5, "S2":5, "S3":8, "PR2":8} mit den Schwellenwerten für die N3-Zuordnung
    """
    
    # drop Summe und N:
    # index und Diskussion dropen und den Code, falls vorhanden (bei M nicht vorhanden, bei S und PR schon):
    try: 
        df = df.drop(columns=[code])
    except: 
        nonsense = 42 
    try: 
        df = df.drop(columns=["SUMME"])
    except: 
        nonsense = 42
    try: 
        df = df.drop(columns=["N = Dokumente/Sprecher"])
    except: 
        nonsense = 42
    try: 
        df = df.drop(columns=[" "])
    except: 
        nonsense = 42

    # Nun die Niveauzuordnung vorbereiten:
    cols = df.columns.values
    niveaus = np.zeros(len(cols))

    for i in np.arange(len(cols)):
        
        # zunächst die Nivaus aus den Kategorienbezeichnungen aus MaxQDA extrahieren. 
        niveaus[i] = extract_number(cols[i])           

        # multiply each columns with the corresponding niveau:
        df_niv = df * niveaus[np.newaxis, :]

        # Compute the maximum of each row (this is the niveau alrealy, exept for code in codes_N3_as_addition, this is S1, S2, S3, PR2)
        max_values = df_niv.max(axis=1)
        
        # Compute the sum of the "Niveaupoints": in some cases it is needed for the N3-attribution:
        sum_values = df_niv.sum(axis=1)

        # Add the new row to the DataFrame
        df_niv['Niveau'] = max_values
        
        # als Hilfsspalte auch die Niveausumme hinzufügen (später wieder löschen):
        df_niv['Niveausumme'] = sum_values
        
        
    # For several codes, N3 is set if there several categories of N2 are triggered:
    if code in N3_Schwellen:
        schwelle = N3_Schwellen[code]
        df_niv['Niveau'] = df_niv.apply(lambda row: 3 if row['Niveausumme'] >= schwelle else row['Niveau'], axis=1)
    else:
        nonsense = 42
        
    # lösche nun die Spalte "Niveausumme" wieder:
    df_niv = df_niv.drop(columns=["Niveausumme"])
    
    return df_niv

#####################################################################################
##################### Kappa berechnen aus zwei dataframes     #######################
#####################################################################################

def calc_kappa(df_Zoho, df_Inte):
    # Berechne nun kappa für die Kategorien: 
    # berechne a, die Anzahl Codes, die in beiden df vergeben wurden:
    df_sum_a = df_Zoho + df_Inte
    a = (df_sum_a == 2).sum().sum() # Idee: in beiden df ist der Code vergeben
    a = a*2 # eingefügt, weil MaxQDA das auch so berechnet, wenn man angibt, dass die Segmente beider Personen berücksichtigt werden sollen: dann bedeutet eine Übereinstimmende Kodierung nämlich, dass die Codierung von Person A auch bei Person B ist UND umgekehrt. MaxQDA zählt das dann doppelt und ich übernehme das hier.


    # berechne b, die Anzahl an Codes, die in df_Zoho vergeben ist, aber nicht in df_Inte:
    df_sum_b = df_Zoho*2 - df_Inte
    b = (df_sum_b == 2).sum().sum()

    # analog c (Code in df_Inte gegeben, aber nicht in df_Zoho)
    df_sum_c= df_Inte*2 - df_Zoho
    c = (df_sum_c == 2).sum().sum()

    # Anzahl Codes
    n_codes = len(df_Zoho.columns.values)

    # Wahrscheinlichkeit für beobachtete Übereinstimmung:
    P_observed = a / (a + b + c)

    # Wahrscheinlichkeit für zufällige Übereinstimmung:
    P_chance = 1 / n_codes

    # kappa:
    kappa_kat = (P_observed - P_chance) / (1 - P_chance)
    
    return kappa_kat, a, b, c, n_codes, P_observed, P_chance



def calc_kappa_niveau(col1, col2):
    
    # zuerst die Koeffizieneten a, b, c berechnen:
    a = (col1 == col2).sum()
    b = (col1 > col2).sum()
    c = (col1 < col2).sum()
    a = a*2 # eingefügt, weil MaxQDA das auch so berechnet, wenn man angibt, dass die Segmente beider Personen berücksichtigt werden sollen: dann bedeutet eine Übereinstimmende Kodierung nämlich, dass die Codierung von Person A auch bei Person B ist UND umgekehrt. MaxQDA zählt das dann doppelt und ich übernehme das hier.


    # Anzahl Codes
    n_codes = 4 # vier Niveaus möglich

    # Wahrscheinlichkeit für beobachtete Übereinstimmung:
    P_observed = a / (a + b + c)

    # Wahrscheinlichkeit für zufällige Übereinstimmung:
    P_chance = 1 / n_codes

    # kappa:
    kappa_Niv = (P_observed - P_chance) / (1 - P_chance)
    
    return kappa_Niv, a, b, c, n_codes, P_observed, P_chance



#####################################################################################
##################### Funktionen für Schritt 1 der Auswertung #######################
#####################################################################################

def zaehle_geschlecht(df):

    Geschlecht_Daten = df["Geschlecht"].values

    m = 0
    w = 0
    d = 0
    kA = 0
    tot = 0
    for i in Geschlecht_Daten:
        tot = tot + 1
        if i == "männlich":
            m = m+1
        elif i == "weiblich":
            w = w+1
        elif i == "divers":
            d = d+1
        elif i == "keine Angabe" or i == "XXX (leeres Feld)":
            kA = kA + 1
        else:
            print(i)


    print("m: " + str(m))
    print("w: " + str(w))
    print("d: " + str(d))
    print("kA: " + str(kA))
    print("tot: " + str(tot))
    print("m+w+d+kA: " + str(m+w+d+kA))

    return m, w, d, kA, tot

def replace_word_by_int_for_Likert(df):
    #DataFrame.replace(to_replace=None, value=_NoDefault.no_default, *, inplace=False, limit=None, regex=False, method=_NoDefault.no_default)
    df = df.replace(to_replace="Stimme überhaupt nicht zu.", value=1)
    df = df.replace(to_replace="sehr unsicher", value=1)
    df = df.replace(to_replace="gar nicht", value=1)
    
    df = df.replace(to_replace="Stimme eher nicht zu.", value=2)
    df = df.replace(to_replace="eher unsicher", value=2)
    df = df.replace(to_replace="eher nicht", value=2)
    
    df = df.replace(to_replace="mittel", value=3)
    
    df = df.replace(to_replace="Stimme eher zu.", value=4)
    df = df.replace(to_replace="eher sicher", value=4)
    df = df.replace(to_replace="eher", value=4)
    
    df = df.replace(to_replace="Stimme voll zu.", value=5)
    df = df.replace(to_replace="sehr sicher", value=5)
    df = df.replace(to_replace="sehr", value=5)
    
    df = df.replace(to_replace="keine Antwort", value=0)
    
    return df
    
    
### Funktion, die drei Dateinamen nimmt:
# aus dem ersten wird die erste Zeile als header genommen
# aus dem zweiten wird das Dataframe eingelesen ohne die ersten beiden Zeilen. Es wird der header aus der ersten Datei hinzugefügt
# in den dritten Dateinamen wird das Ergebnis als Exceltabelle und pkl geschrieben
    
def replace_header(xlsx_with_header, csv_with_data, name_for_result, Ordner_Rohdaten, Ordner_Zwischendaten):
    Ordnername_keys_header = "keys_header"
    #Ordner_Rohdaten = "2023-02-07_DatenZohoFinal"
    #Ordner_Zwischendaten = "DatenArbeit"
    
    # open header-file and convert header to list:
    df_header = pd.read_excel(Ordnername_keys_header + "/" + xlsx_with_header)
    header    = list(df_header.columns.values)
    
    # read data-frame without first two lines:
    df_data = pd.read_csv(Ordner_Rohdaten + "/" + csv_with_data, index_col = False, skiprows=2)#, columns = header)
    df_data.columns = header
    
    # remove last two columns (as without information):
    df_data.drop(columns=df_data.columns[-2:], axis=1, inplace=True)
    
    # replace words to numbers for Likert-Items
    df_data = replace_word_by_int_for_Likert(df_data)
    
    # save to file:
    df_data.to_excel(Ordner_Zwischendaten + "/" + name_for_result + ".xlsx")
    
    print("done.")
    return df_data


def remove_blacklist_from_df(df, blacklist):
    """ Removes all rows from the dataframe df, that have a Code listed in the blacklist. Returns the shortened dataframe."""
    
    rows_blacklist = df.loc[df["Code"].isin(blacklist)].index
    df_clean = df.drop(rows_blacklist)
    
    return df_clean    
    
    
def create_P_and_X_key_df(df_pre, df_post):
    
    # Dataframes "P-key" und "X-key" vorbereiten für die Zuordnung der Codes:

    # alt: columntitels = ["Code", "Code_V2", "Code_V3", "Code_V4", "Nr", "Pre", "BU1", "BU2", "BU3Te", "BU3Vi", "BU4", "BU5", "BU6", "Post"]
    columntitels = ["Code", "Nr", "Pre", "BU1", "BU2", "BU3Te", "BU3Vi", "BU4", "BU5", "BU6", "Post"]

    # Create empty DataFrames:
    df_P_key = pd.DataFrame(columns = columntitels)
    df_X_key = pd.DataFrame(columns = columntitels)
    
    # nun die Zeilen hinzufügen: 
    zaehler_P = 0
    zaehler_X = 0
    for c in df_pre['Code'].values:
        if c in df_post['Code'].values:
            df_P_key = df_P_key.append({"Code":c, "Nr": zaehler_P, "Pre":1}, ignore_index=True)
            zaehler_P = zaehler_P + 1

        else:
            df_X_key = df_X_key.append({"Code":c, "Nr": zaehler_X, "Pre":1}, ignore_index=True)
            zaehler_X = zaehler_X + 1

    return df_P_key, df_X_key
        

def create_df_with_remaining_codes_from_posttest(df_pre, df_post):
    
    rows_in_post_with_P_Nr = df_post.loc[df_post["Code"].isin(df_pre["Code"])].index
    print(rows_in_post_with_P_Nr)
    df_post_uebrige_codes = df_post.drop(rows_in_post_with_P_Nr)
    df_post_uebrige_codes = df_post_uebrige_codes[["Code"]]
    #df = df[['col2', 'col6']] --> drops all columns exept col2, col6
    nr_remaining_codes = df_post_uebrige_codes.shape[0]
    
    return df_post_uebrige_codes, nr_remaining_codes


def gib_zeile_von_eintrag_in_spalte(df, eintrag, spalte):
    zeile = df.loc[df[spalte] == eintrag].index[0]
    return zeile


def codes_small(df):
    df_return = df
    len_df = df.shape[0]
    for i in range(len_df):
        Codes_orig = df.at[i, "Code"]
        Codes_small = str(Codes_orig).lower()
        
        df_return.at[i, "Code"] = Codes_small
        
    return df_return


####################### Funktionen für weitere Auswertung #################


def concat_pre_post(df_pre, df_post):
    df_pre.columns = df_pre.columns.str.replace('_pre', '')
    df_post.columns = df_post.columns.str.replace('_post', '')
    df_beide = pd.concat([df_pre, df_post], ignore_index=True)
    return df_beide

def concat_KorrSicherheit(df_pre, df_post, code):
    # M1Niv_post 	M1_post_Sicherheit
    df_pre.columns = df_pre.columns.str.replace(code + "Niv_pre", 'Niveau')
    df_pre.columns = df_pre.columns.str.replace(code + "_pre_Sicherheit", 'Sicherheit')
    df_post.columns = df_post.columns.str.replace(code + "Niv_post", 'Niveau')
    df_post.columns = df_post.columns.str.replace(code + "_post_Sicherheit", 'Sicherheit')
    
    df_beide = pd.concat([df_pre, df_post], ignore_index=True)
    return df_beide

def df_corr(df):
    # Die Korrelationsmatrix berechnen:
    df_corr = df.corr(method='spearman', min_periods=2, numeric_only=True)
    # Nun auf zwei Nachkommastellen runden (nur Ausgabe, Werte gibt es weiterhin)
    df_corr_styled = df_corr.style.format(precision=2)
    # Diese Matrix als Latex-Tabelle drucken:
    print(df_corr_styled.to_latex())
    #die volle Korrelationsmatrix zurückgeben:
    return df_corr

def ca(df):
    return pg.cronbach_alpha(data=df)[0]
    #return cronbach_alpha(df) --> Berechnung mit der Implementation der Website s.o.


def df_print_latex(df, precision_digits):
    df_styled = df.style.format(precision=precision_digits)
    print(df_styled.to_latex())






########################## weitere #########################




# Eigene Funktionen für die 4K Auswertung:

# die Werte invertieren:
def invert_LikertScale(df): # doch nciht genutzt wegen eleganterer Alternative
    df = df.replace(to_replace=1, value=101)
    df = df.replace(to_replace=2, value=102)
    
    df = df.replace(to_replace=5, value=1)
    df = df.replace(to_replace=4, value=2)
    
    df = df.replace(to_replace=101, value=5)
    df = df.replace(to_replace=102, value=4)
    
    # alternativ und wohl eleganter: return = 6 - df
    
    return df

def assign_4K_columns(df):
        
    # see https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
    list_CritThin = ["CritThink-1", "CritThink-2", "CritThink-3", "CritThink-4", "CritThink-5", "CritThink-6", "CritThink-7", "CritThink-8", "CritThink-9", "CritThink-10", "CritThink-11"]
    list_Collab   = ["Collab-1", "Collab-2", "Collab-3", "Collab-4", "Collab-5", "Collab-6", "Collab-7", "Collab-8"]
    list_Commu    = ["Commu-1", "Commu-2", "Commu-3", "Commu-4", "Commu-5"]
    list_Creativ  = ["Creativ-1", "Creativ-2", "Creativ-3", "Creativ-4", "Creativ-5", "Creativ-6", "Creativ-7"]
        
    df = df.assign(CritThink = df[list_CritThin].mean(axis=1))
    df = df.assign(Collab    = df[list_Collab].mean(axis=1))
    df = df.assign(Commu     = df[list_Commu].mean(axis=1))
    df = df.assign(Creativ   = df[list_Creativ].mean(axis=1))

    df = df.assign(alle4K = df.mean(axis=1))
    
    return df

def assign_4K_columns_Janina(df):
    """
    In Janinas Version des Testes waren ein paar Items nicht drin. Deswegen werden diese auch hier ausgelassen. Es fehlen: CritThink-8,9,10; Collab-3,7; Creativ-4
    """

    # see https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
    list_CritThin = ["CritThink-1", "CritThink-2", "CritThink-3", "CritThink-4", "CritThink-5", "CritThink-6", "CritThink-7", "CritThink-11"]
    list_Collab   = ["Collab-1", "Collab-2", "Collab-4", "Collab-5", "Collab-6", "Collab-8"]
    list_Commu    = ["Commu-1", "Commu-2", "Commu-3", "Commu-4", "Commu-5"]
    list_Creativ  = ["Creativ-1", "Creativ-2", "Creativ-3", "Creativ-5", "Creativ-6", "Creativ-7"]
            
    df = df.assign(CritThink = df[list_CritThin].mean(axis=1))
    df = df.assign(Collab    = df[list_Collab].mean(axis=1))
    df = df.assign(Commu     = df[list_Commu].mean(axis=1))
    df = df.assign(Creativ   = df[list_Creativ].mean(axis=1))

    df = df.assign(alle4K = df.mean(axis=1))
    
    return df


# eine Funktion, die für jedes 4K einzeln aufgerufen werden kann und dann die Verteilungen von pre und post 
# in ein Histogram plottet und einen zweiseitigen t-Test dazwischen berechnet:

def tTest_Cohensd_plotHistogram(df1, df2, spalte, spalteDE, label1, label2, pfad, titel_diagram_Anfang):
    
    # Herausholen der relevanten Spalte:
    df1_spalte = df1[spalte]
    df2_spalte = df2[spalte]
    
    # für plot:
    df_plot = pd.concat([df1_spalte, df2_spalte], ignore_index=True, axis=1)
    df_plot.columns = [label1, label2]
    
    # Wenn ein Wert nan ist, wird die ganze Zeile (=Person) herausgeworfen:
    df_plot = df_plot.dropna(axis=0, how='any')
    
    # für stat-Tests, np.hist und Titel
    data1 = df_plot[label1].to_numpy()
    data2 = df_plot[label2].to_numpy()
    
    N = len(data2)
    
    # Cohens d:
    d = cohens_d(data2[~np.isnan(data2)], data1[~np.isnan(data1)]) # mit den Daten so herum, ist eine Vergrößerung von pre zu post als positiver d-Wert angegeben 
    str_d = str(np.round(d,3))
    
    # t-Test (für zwei verbundene Stichproben)
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    # scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagate', alternative='two-sided', *, keepdims=False)
    res    = ttest_rel(data1, data2, alternative='less')
    stat   = res.statistic
    p_Wert = res.pvalue
    if p_Wert < 0.001:
        str_p_Wert = '%.3E' % Decimal(p_Wert)
    else:
        str_p_Wert = str(np.round(p_Wert, 4))
    
    # Plot histogram with relative frequencies on y-axis:
    Likert_Labels_voll = ["Stimme überhaupt nicht zu.", "Stimme eher nicht zu.", "mittel", "Stimme eher zu.", "Stimme voll zu."]
    Likert_Labels_kurz = ["1: überhaupt nicht", "2: eher nicht", "3: mittel", "4: eher", "5: voll"]
    Likert_num    = ["1", "2", "3", "4", "5"]
    
    sns.set(style="darkgrid")
    #fig = sns.histplot(data=df_plot, bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color="skyblue", multiple = "dodge", shrink = 0.3, discrete = True, stat="percent", common_norm=False)
    fig = sns.histplot(data=df_plot, color="skyblue", multiple = "dodge", shrink = 0.8, stat="percent", common_norm=False) #binwidth=0.2, , discrete = True
    #fig.set_xticks([1, 2, 3, 4, 5], labels=Likert_Labels_kurz)
    fig.set_title(titel_diagram_Anfang + spalteDE + ", \n N  = " + str(N) + ", p (t-Test): " + str_p_Wert + ", Cohens d: " + str_d)
    plt.tight_layout()
    #plt.savefig(pfad + spalte + ".pdf")
    plt.savefig(pfad + spalte + ".svg")
    plt.savefig(pfad + spalte + ".png")
    plt.show()
    plt.close()
    
    return stat, p_Wert, d, N, np.mean(data1), np.mean(data2), np.std(data1), np.std(data2)


# Die Funktionen aus Schritt 2a QuantAuswertung Kategorien v16:

def read_dataframe_data(Ordner, dateiname):
    """
    Funktion, die eine Datendatei als Pandas Dataframe einliest und transponiert
    """
    df = pd.read_excel(Ordner + "/" + dateiname)
    df = df.transpose()
    df.columns = df.iloc[0].astype(str)
    df = df.drop(["Unnamed: 0"])
    df = df.drop(["Total"])
    df.index = df.index.str.strip()
    return df

def add_Punkte(df, dateiname_KatPunkte, Ordner):
    """
    Fügt dem Dataframe df die Spalte Punkte hinzu und trägt dort die Punkte jeder Person ein, 
    als Summe über die vergebenen Kategorien je gewichtet mit den Kategoriepunkten in dem Dataframe df_KatPunkte.
    return: das Eingangs-Dataframe mit der zusätzlichen Spalte Punkte.
    """
    
    # Einlesen des Schlüssels Kategorien - Punkte und Hinzufügen der Spaltenüberschriften:
    df_KatPunkte = pd.read_excel(Ordner + "/" + dateiname_KatPunkte, header=None)
    df_KatPunkte.columns = ["KatNamen", "Punkte"]
    #print(df_KatPunkte)
    
    # Hinzufügen der Spalten mit den gewichteten Kategoriensetzungen im Dataframe df: 
    zaehler = 0
    for kat in df_KatPunkte["KatNamen"]:
        #print(kat)
        zeile = df_KatPunkte.loc[df_KatPunkte["KatNamen"] == kat].index[0]
        #print(zeile)
        punkte = df_KatPunkte.at[zeile, "Punkte"]
        
        kat_new = str(kat) + "_neu"
        #print(kat)
        df = df.assign(new = df[kat]*punkte)
        df = df.rename(columns={"new": kat_new})
        zaehler = zaehler + 1

    # Hinzufügen der Spalte "Punkte" in df (falls es diese nicht doch schon gibt):
    df = df.assign(Punkte = df.iloc[:, -1]*0)
    
    # Erstellung des Dataframes df_neu, das nur die neuen Spalten enthält. Dies dient nur dazu später einfacher darüber zu summieren:
    df_neu = df.iloc[:, -(zaehler+1):]
    
    Summenspalte = df_neu.sum(axis=1)
    
    df["Punkte"]= df_neu.sum(axis=1)
   
    return df

def Punkte_zu_Niveau(p, schluessel_PunkteNiveau):
    a = schluessel_PunkteNiveau[0]
    b = schluessel_PunkteNiveau[1]
    c = schluessel_PunkteNiveau[2]
    d = schluessel_PunkteNiveau[3]
    
    # z.B.: schluessel_PunkteNiveau_S1 = [-500, 1, 2, 5]
    # hierbei für [a, b, c, d] und eine Punktzahl P:
    # keine Zuordnung: P < a  , sprich: "ein Wert unter a Punkten bedeutet, dass keine Zuordnung möglich ist"
    # N0: a <= P < b  , sprich: "ab a Punkten hat man N0 erreicht"
    # N1: b <= P < c  , sprich: "ab b Punkten hat man N1 erreicht"
    # N2: c <= P < d  , sprich: "ab c Punkten hat man N2 erreicht"
    # N3: d <= P      , sprich: "ab d Punkten hat man N3 erreicht"
    
    if p < a:
        niveau = 4    
    elif a <= p < b:
        niveau = 0
    elif b <= p < c:
        niveau = 1
    elif c <= p < d:
        niveau = 2
    elif d <= p:
        niveau = 3
    else:
        print("Upps, hier ist etwas schief gegangen.")
    
    return niveau


def add_Niveaus(df, schluessel_PunkteNiveau):
    """
    Fügt dem Dataframe df die Spalte Niveau hinzu und trägt dort das Niveau jeder Person ein, 
    entsprechend dem Array "schluessel_PunkteNiveau", getestet mit der Funktion "Punkte_zu_Niveau".
    return: das Eingangs-Dataframe mit der zusätzlichen Spalte Niveau.
    """
      
    # Hinzufügen der Spalte "Niveau" in df (falls es diese nicht doch schon gibt):
    df = df.assign(Niveau = df.iloc[:, -1]*0+99) # hierhier
    
    # Iterating over one column - `f` is some function that processes your data
    #result = [f(x) for x in df['col']]
    df["Niveau"] = [Punkte_zu_Niveau(p, schluessel_PunkteNiveau) for p in df["Punkte"]]
       
    return df

def sort_df_PX(df):
    """Ergänzt eine Spalte >Sortierung< in df, dessen Einträge aus dem Index generiert wurden, wobei:
    if Index beginnt mit P: int in Sortierung = P-Nummer
    if Index beginnt mit X: int in Sortierung = X-Nummer + 1000
    else: Fehlermeldung als print
    """
    N = df.shape[0]

    df["hilfszeile"] = df.index
    df["Sortierung"] = None

    for key in df.index.values:
        ind = df.at[key, "hilfszeile"] # ind als Kurzform zu index an der Stelle
        
        first_char = ind[0]
       
        if first_char == "X":
            if len(ind) == 2:
                str_num = ind[1]
            elif len(ind) == 3:
                str_num = ind[1] + ind[2]
            elif len(ind) == 4:
                str_num = ind[1] + ind[2] + ind[3]
            else:
                print("Es gibt einen Fehler beim Buchstabenzählen bei " + ind)
            
            num = int(str_num)
            num = num + 1000 # sortiert die X-Personen nach hinten, 1000 größer als jede Zahl mit drei Stellen. Diese Routine funktioniert also nur bei Stichproben mit N<1000. Dies ist bei uns erfüllt.

        elif first_char == "P":
            if len(ind) == 2:
                str_num = ind[1]
            elif len(ind) == 3:
                str_num = ind[1] + ind[2]
            elif len(ind) == 4:
                str_num = ind[1] + ind[2] + ind[3]
            else:
                print("Es gibt einen Fehler beim Buchstabenzählen bei " + ind)
            
            num = int(str_num) # P-Nummern nicht ändern

        else:
            print("Es gibt einen Fehler mit dem ersten Buchstaben bei " + ind)

        df.at[key, "Sortierung"] = num
    
    df = df.drop(columns = ["hilfszeile"])
    
    return df

def smart_merge_df(df1, df2, spalte, label1, label2):

    len_df1_1 = df1.shape[0]
    len_df2_1 = df2.shape[0]

    # lösche im Index alle "pre" und "post":
    df1.index = df1.index.str.rstrip('pre')
    df2.index = df2.index.str.rstrip('post')

    # remove all rows from df1, which are NOT in df2:
    df1 = df1[df1.index.isin(df2.index)]
    # remove all rows from df2, which are NOT in df1:
    df2 = df2[df2.index.isin(df1.index)]
    
    # Sortieren der DataFrames (manchmal nicht nötig, machmal aber schon):
    df1 = sort_df_PX(df1)                   #Spalte erstellen, die zum Sortieren genutzt werden kann
    df1 = df1.sort_values("Sortierung")     #Das df nach dieser Spalte sortieren
    df2 = sort_df_PX(df2)        
    df2 = df2.sort_values("Sortierung")   
    
    # Überprüfe, ob die Personen richtig zugeordent sind:
    for i in range(len(df2.index.values)):
        if df1.index.values[i] != df2.index.values[i]:
            print("Zuordnungsfehler bei i = " + str(i) + ", " +label1 + " : " + df1.index.values[i] + ", " + label2 + " : " + df2.index.values[i])
        else:
            a = 42
  

    # Ausgabe zur Kontrolle:
    len_df1_2 = df1.shape[0]
    len_df2_2 = df2.shape[0]

    removed_in_df1 = len_df1_1 - len_df1_2
    removed_in_df2 = len_df2_1 - len_df2_2
    check = len_df2_2 - len_df1_2 # should be zero
    if check != 0:
        logging.info("Upps, da ist was schief gegangen! Die beiden Datenreihen sind trotz Abgleich nicht gleich lang.")
        print("Upps, da ist was schief gegangen! Die beiden Datenreihen sind trotz Abgleich nicht gleich lang.")
    else:
        print("Länge der Datenreihen nach Abgleich: " + str(len_df1_2))
        print("Anzahl in df1 gelöschter Zeilen beim Ablgleich: " + str(removed_in_df1))
        print("Anzahl in df2 gelöschter Zeilen beim Ablgleich: " + str(removed_in_df2))
        logging.info("Länge der Datenreihen nach Abgleich: " + str(len_df1_2))
        logging.info("Anzahl in df1 gelöschter Zeilen beim Ablgleich: " + str(removed_in_df1))
        logging.info("Anzahl in df2 gelöschter Zeilen beim Ablgleich: " + str(removed_in_df2))

    # fasse die beiden Niveau-Spalten in einem (neuen) Dataframe zusammen. Dies ist praktisch für die weitere Handhabung.
    df = pd.DataFrame(columns=[label1, label2], index = df1.index)
    df[label1] = df1[spalte].values
    df[label2] = df2[spalte].values


    # hier: entferne alle Zeilen, in denen ein Eintrag = 4 ist:
    df.drop(df.loc[df[label1]==4].index, inplace=True)
    df.drop(df.loc[df[label2]==4].index, inplace=True)

    # noch eine Kontrollausgabe:
    len_df_3 = df.shape[0]
    anz_zeilen_ungültig = len_df2_2 - len_df_3
    print("Anzahl in df gelöschter Zeilen, weil mindestens die pre oder post-Antwort nicht zuordbar war: " + str(anz_zeilen_ungültig))
    logging.info("Anzahl in df gelöschter Zeilen, weil mindestens die pre oder post-Antwort nicht zuordbar war: " + str(anz_zeilen_ungültig))

    return df, df1, df2, removed_in_df1, removed_in_df2, anz_zeilen_ungültig

# df, df1, df2, removed_in_df1, removed_in_df2, anz_zeilen_ungültig = smart_merge_df(df1, df2, spalte, label1, label2)


def cohens_d(x,y):
    # see https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
    # and see: https://de.wikipedia.org/wiki/Effektst%C3%A4rke#Cohens_d
    cohens_d = (np.mean(x) - np.mean(y)) / (np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2))
    return cohens_d


def Wilcoxon_Effektstaerke_plotHistogram(df, label1, label2, pfad, titel_diagram_Anfang, TestEinZweiSeitig):
    """
    TestEinZweiSeitig: Alternativ-Hypothese des Wilcoxon-Tests, aus Dokumentation mit d = x - y:
            ‘two-sided’: the distribution underlying d is not symmetric about zero.
            ‘less’: the distribution underlying d is stochastically less than a distribution symmetric about zero.
            ‘greater’: the distribution underlying d is stochastically greater than a distribution symmetric about zero.
            Wenn y = post also größer ist als x = pre, ist d = pre - post also negativ. Wir nehmen "less" dafür. 
    """
    
    # Herausholen der x-,y-Daten:
    x_data = df[label1].values
    y_data = df[label2].values
    N = df.shape[0] # N a priori bei beiden gleich
    
    #Wilcoxon Test:
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    res    = wilcoxon(x = df[label1], y=df[label2], alternative=TestEinZweiSeitig) # "mode" ist alt, in der aktuellen Version heißt es "method"
    stat   = res.statistic
    p_Wert = res.pvalue
    
    print("Wilcoxon:")
    print(stat, p_Wert)
    str_p_Wert = '%.3E' % Decimal(p_Wert)
    
    # Pearsons R:
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    result        = pearsonr(x = x_data, y=y_data)
    R_Pearson     = result[0]
    #str_R_Pearons = str(np.round(R,3))
    
    # Spearman R:
    result_spearmanr = spearmanr(a = x_data, b=y_data)
    print(result_spearmanr)
    R_Spearman = result_spearmanr.correlation
    
    # Cohens d:
    d_Cohen = cohens_d(x_data,y_data)
    #str_d = str(np.round(d,3))
    
   
    # Plot histogram with relative frequencies on y-axis:
    sns.set(style="darkgrid")
    fig = sns.histplot(data=df, bins = [-0.5, 0.5, 1.5, 2.5, 3.5], color="skyblue", multiple = "dodge", shrink = 0.3, discrete = True, stat="percent", common_norm=False)#, legend=True)
    fig.set_xticks([0, 1, 2, 3], labels=['N0','N1','N2','N3'])
    fig.set_title(titel_diagram_Anfang + "\n N = " + str(N) + ", p (Wilcoxon): " + str_p_Wert)
    plt.savefig(pfad + "_rel.svg")
    #plt.savefig(pfad + "_rel.png")
    plt.savefig(pfad + "_rel.pdf")
    plt.show()
    plt.close()
    
    hist_abs_x, hist_rel_x, quartil_unten_x, median_x, quartil_oben_x = berechne_Histwerte_Quantile(x_data)
    hist_abs_y, hist_rel_y, quartil_unten_y, median_y, quartil_oben_y = berechne_Histwerte_Quantile(y_data)
    
    # schreibe alle Werte in eine Excel-Tabelle:
    alles_zusammen = pd.concat([pd.Series(hist_abs_x),         pd.Series(hist_abs_y),           pd.Series(hist_rel_x),          pd.Series(hist_rel_y),          pd.Series(N),   pd.Series(stat),   pd.Series(p_Wert), pd.Series(d_Cohen), pd.Series(R_Pearson), pd.Series(R_Spearman)    , pd.Series(quartil_unten_x),        pd.Series(median_x),        pd.Series(quartil_oben_x),          pd.Series(quartil_unten_y), pd.Series(median_y), pd.Series(quartil_oben_y)       ], axis=1, ignore_index=True)
    alles_zusammen.columns = ["Hist-Werte (abs) in " + label1, "Hist-Werte (abs) in " + label2, "Hist-Werte (rel) in " + label1, "Hist-Werte (rel) in " + label2, "N "      ,  "stat (Wilcoxon)", "p-Wert (Wilcoxon)", "Cohens d",        "Pearons R",          "Spearman R",            "25Proz/unteres Quartil " + label1, "Median " + label1, "75Proz/oberes Quartil " + label1, "25Proz/unteres Quartil " + label2, "Median " + label2 , "75Proz/oberes Quartil " + label2]
    alles_zusammen.to_excel(pfad + "_Ergebnisse.xlsx", index=False)
    

    return stat, p_Wert, R_Pearson, R_Spearman, d_Cohen, N, hist_abs_x, hist_rel_x, quartil_unten_x, median_x, quartil_oben_x, hist_abs_y, hist_rel_y, quartil_unten_y, median_y, quartil_oben_y 


def berechne_Histwerte_Quantile(data):
    # berechne auch die Werte des Histograms:
    N = len(data)
    hist_abs, bin_edges = np.histogram(data, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
    hist_rel = hist_abs/N*100
    
    # Median und Quartile:
    median        = np.median(data)
    quartil_oben  = np.quantile(data, 0.75)
    quartil_unten = np.quantile(data, 0.25)
        
    return hist_abs, hist_rel, quartil_unten, median, quartil_oben

# hist_abs, hist_rel, quartil_unten, median, quartil_oben = berechne_Histwerte_Quantile(data)

def berechne_Histwerte_Quantile_Likert(data):
    # berechne auch die Werte des Histograms:
    N = len(data)
    hist_abs, bin_edges = np.histogram(data, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    hist_rel = hist_abs/N*100
    
    # Median und Quartile:
    median        = np.median(data)
    quartil_oben  = np.quantile(data, 0.75)
    quartil_unten = np.quantile(data, 0.25)
        
    return hist_abs, hist_rel, quartil_unten, median, quartil_oben


def MannWhitneyU_Effektstaerke_plotHistogram(df1, df2, spalte, label1, label2, pfad, titel_diagram_Anfang):

    # Herausholen der relevanten Spalte:
    df1_spalte = df1[spalte]
    df2_spalte = df2[spalte]
    df1_spalte.index = np.arange(df1_spalte.shape[0])
    df2_spalte.index = np.arange(df2_spalte.shape[0])

    # hier: Zeilen mit "4" entfernen:
    df1_spalte.drop(df1_spalte.loc[df1_spalte==4].index, inplace=True)
    df2_spalte.drop(df2_spalte.loc[df2_spalte==4].index, inplace=True)

    # für stat-Tests, np.hist und Titel
    x_data = df1_spalte.values
    y_data = df2_spalte.values
    N1 = len(x_data)
    N2 = len(y_data)

    # für plot:
    df_plot = pd.concat([df1_spalte, df2_spalte], ignore_index=True, axis=1)
    df_plot.columns = [label1, label2]


    #MannWhitneyU:
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    res    = mannwhitneyu(x = x_data, y=y_data, alternative='two-sided')
    stat   = res.statistic
    p_Wert = res.pvalue
    #z_Wert = zstatistic

    if p_Wert < 0.001:
        str_p_Wert = '%.3E' % Decimal(p_Wert)
    else:
        str_p_Wert = str(np.round(p_Wert, 5))

    # Cohens d:
    d = cohens_d(x_data,y_data)
    str_d = str(np.round(d,3))

    # Plot histogram with relative frequencies on y-axis:
    sns.set(style="darkgrid")
    fig = sns.histplot(data=df_plot, bins = [-0.5, 0.5, 1.5, 2.5, 3.5], color="skyblue", multiple = "dodge", shrink = 0.3, discrete = True, stat="percent", common_norm=False)#, legend=True)
    fig.set_xticks([0, 1, 2, 3], labels=['N0','N1','N2','N3'])
    fig.set_title(titel_diagram_Anfang + ", \n N " + label1 + " = " + str(N1) + ", N " + label2 + " = " + str(N2) + ", p (Mann-Whitney U): " + str_p_Wert)
    plt.savefig(pfad + "_rel.svg")
    plt.savefig(pfad + "_rel.pdf")
    plt.show()
    plt.close()
    
    hist_abs_x, hist_rel_x, quartil_unten_x, median_x, quartil_oben_x = berechne_Histwerte_Quantile(x_data)
    hist_abs_y, hist_rel_y, quartil_unten_y, median_y, quartil_oben_y = berechne_Histwerte_Quantile(y_data)
    
    # schreibe alle Werte in eine Excel-Tabelle:
    alles_zusammen = pd.concat([pd.Series(hist_abs_x),         pd.Series(hist_abs_y),           pd.Series(hist_rel_x),          pd.Series(hist_rel_y),           pd.Series(N1),   pd.Series(N2), pd.Series(stat),        pd.Series(p_Wert),          pd.Series(d), pd.Series(quartil_unten_x), pd.Series(median_x),          pd.Series(quartil_oben_x),          pd.Series(quartil_unten_y),        pd.Series(median_y), pd.Series(quartil_oben_y)       ], axis=1, ignore_index=True)
    alles_zusammen.columns = ["Hist-Werte (abs) in " + label1, "Hist-Werte (abs) in " + label2, "Hist-Werte (rel) in " + label1, "Hist-Werte (rel) in " + label2, "N " + label1, "N " + label2, "stat (Mann Whitney U)", "p-Wert (Mann Whitney U)", "Cohens d", "25Proz/unteres Quartil "      + label1, "Median " + label1, "75Proz/oberes Quartil " + label1, "25Proz/unteres Quartil " + label2, "Median " + label2, "75Proz/oberes Quartil " + label2]
    alles_zusammen.to_excel(pfad + "_Ergebnisse.xlsx", index=False)
    
    return stat, p_Wert, d, N1, N2

# stat, p_Wert, d, N1, N2 = MannWhitneyU_Effektstaerke_plotHistogram(df1, df2, spalte, label1, label2, pfad, titel_diagram_Anfang)



def Personen_aufteilen_ViTe(df_gesamt, dateiname_vergleich):
    
    df_vergleich = read_dataframe_data(Ordner_Quelle, dateiname_vergleich)
    
    # lösche im Index alle "pre" und "post":
    df_gesamt.index = df_gesamt.index.str.rstrip('pre')
    df_gesamt.index = df_gesamt.index.str.rstrip('post')
    df_vergleich.index = df_vergleich.index.str.rstrip('pre')
    df_vergleich.index = df_vergleich.index.str.rstrip('post')

    # Lösche alle Zeilen, die nicht im Vergleichs-Dataframe sind:
    df_cut = df_gesamt[df_gesamt.index.isin(df_vergleich.index)]
    
    return df_cut


def extract_data_Likert(df1, df2, label1, label2, kuerzel_Vi, kuerzel_Te):
    
    spalte1 = df1[kuerzel_Vi]
    spalte2 = df2[kuerzel_Te]
    
    data1 = spalte1.values
    data2 = spalte2.values
    
    df_plot = pd.concat([spalte1, spalte2], ignore_index=True, axis=1)
    df_plot.columns = [label1, label2]
    
    N1 = len(data1)
    N2 = len(data2)
        
    return df_plot, data1, data2, N1, N2


def tTest_Histogram_BU3_ViTe(df_plot, data1, data2, N1, N2, label1, label2, kuerzel_Titel, pfad):
    """
    Berechnet einen zweiseitigen t-Test zwischen zwei unabhängigen intervallskalierten (normalverteilten) Stichproben und plottet ein Histogramm davon.
    """
    
    # den t-Test (zweiseitig, zwei unabhängige Stichproben) berechnen:
    res    = ttest_ind(data1, data2, alternative='two-sided')
    stat   = res.statistic
    p_Wert = res.pvalue

    if p_Wert < 0.001:
        str_p_Wert = '%.3E' % Decimal(p_Wert)
    else:
        str_p_Wert = str(np.round(p_Wert, 4))

    # Cohens d:
    d = cohens_d(data1, data2)
    str_d = str(np.round(d,4))
    
    # zum späteren Speichern:
    

    # Plot histogram with relative frequencies on y-axis:
    sns.set(style="darkgrid")
    fig = sns.histplot(data=df_plot, bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color="skyblue", multiple = "dodge", shrink = 0.3, discrete = True, stat="percent", common_norm=False)
    fig.set_xticks([1, 2, 3, 4, 5], labels= Likert_Labels)
    fig.set_title(kuerzel_Titel + ", \n N " + label1 + " = " + str(N1) + ", N " + label2 + " = " + str(N2) + ", p (t-Test): " + str_p_Wert + ", Cohens d: " + str_d)
    plt.tight_layout()
    plt.savefig(pfad + "_rel.svg")
    plt.savefig(pfad + "_rel.pdf")
    plt.show()
    plt.close()
    
    hist_abs_x, hist_rel_x, quartil_unten_x, median_x, quartil_oben_x = berechne_Histwerte_Quantile_Likert(data1)
    hist_abs_y, hist_rel_y, quartil_unten_y, median_y, quartil_oben_y = berechne_Histwerte_Quantile_Likert(data2)
    
    # schreibe alle Werte in eine Excel-Tabelle:
    alles_zusammen = pd.concat([pd.Series(hist_abs_x),         pd.Series(hist_abs_y),           pd.Series(hist_rel_x),          pd.Series(hist_rel_y),           pd.Series(N1),   pd.Series(N2), pd.Series(stat),        pd.Series(p_Wert),          pd.Series(d), pd.Series(quartil_unten_x), pd.Series(median_x),          pd.Series(quartil_oben_x),          pd.Series(quartil_unten_y),        pd.Series(median_y), pd.Series(quartil_oben_y)       ], axis=1, ignore_index=True)
    alles_zusammen.columns = ["Hist-Werte (abs) in " + label1, "Hist-Werte (abs) in " + label2, "Hist-Werte (rel) in " + label1, "Hist-Werte (rel) in " + label2, "N " + label1, "N " + label2, "stat (t-Test)", "p-Wert (t-Test)", "Cohens d", "25Proz/unteres Quartil "      + label1, "Median " + label1, "75Proz/oberes Quartil " + label1, "25Proz/unteres Quartil " + label2, "Median " + label2, "75Proz/oberes Quartil " + label2]
    alles_zusammen.to_excel(pfad + "_Ergebnisse.xlsx", index=False)
    
    return stat, p_Wert, d, N1, N2

