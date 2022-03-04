import pandas as pd   
data = pd.read_csv('dados_apos_p_processamento.csv') 
print("Original 'dados_apos_p_processamento.csv' CSV Data: \n") 
print(data) 
data.drop(['Anti-hipertensores' ], inplace=True, axis=1) 
print("\nCSV Data after deleting column:\n") 
print(data)

data.to_csv('resnovos.csv') 



''''

'Casa' 'Inter-Hospitalar' 'Intra-Hospitalar' 'Lar'
 'GrupoDiagn_Cardiovascular' 'GrupoDiagn_Gastrointestinal'
 'GrupoDiagn_Geniturinario' 'GrupoDiagn_Hemato-Oncologico'
 'GrupoDiagn_Musculoesqueletico' 'GrupoDiagn_Neurologico'
 'GrupoDiagn_Outro' 'GrupoDiagn_Respiratorio' 'Local_SU' 'Idade'
 'Interna_Dias' 'SIRS' 'Glicose' 'Sodio' 'Ureia' 'Creatinina' 'PCR' 'pH'
 'Ca_ionizado' 'pCO2' 'pO2' 'HCO3' 'Genero' 'Antidislipidemicos'
 'Antipsicoticos' 'Antidepressores' 'Analgesicos' 'Anticoagulantes'
 'Digitalicos' 'Corticosteroides' 'Outros Med_Presente' 'Alcoolico'



38--------------------

 'Casa' 'Inter-Hospitalar' 'Intra-Hospitalar' 'Lar'
 'GrupoDiagn_Cardiovascular' 'GrupoDiagn_Gastrointestinal'
 'GrupoDiagn_Geniturinario' 'GrupoDiagn_Hemato-Oncologico'
 'GrupoDiagn_Musculoesqueletico' 'GrupoDiagn_Neurologico'
 'GrupoDiagn_Outro' 'GrupoDiagn_Respiratorio' 'Local_SU' 'Idade' 
 'Interna_Dias' 'SIRS' 'Glicose' 'Sodio' 'Ureia' 'Creatinina' 'PCR' '
 pH' 'Ca_ionizado' 'pCO2' 'pO2' 'HCO3' 'Genero' 'Antidislipidemicos' 
 'Antipsicoticos' 'Antidepressores' 'Anti-hipertensores' 'Ansioliticos'
 'Analgesicos' 'Anticoagulantes' 'Digitalicos' 'Corticosteroides'
 'Outros Med_Presente' 'Alcoolico'



Idade' 'Glicose' 'Sodio' 'Ureia' 'Creatinina' 'PCR' 'pH' 'Ca_ionizado' 'pCO2' 
'pO2' 'HCO3' 'Genero' 'Antidislipidemicos' 'Antipsicoticos' 'Antidepressores' 
'Anti-hipertensores' 'Ansioliticos'
 'Analgesicos' 'Outros Med_Presente'


'''