coefs = [('Casa', -0.7173469829801948), ('Inter-Hospitalar', -1.4815082297544002), ('Intra-Hospitalar', -0.6923880746830383), ('Lar', -1.315949612070675), ('GrupoDiagn_Cardiovascular', -1.829437733900101), ('GrupoDiagn_Gastrointestinal', -2.221114581330911), ('GrupoDiagn_Geniturinario', -2.0791736521996516), ('GrupoDiagn_Hemato-Oncologico', -1.1868100747266055), ('GrupoDiagn_Musculoesqueletico', -2.006382454738856), ('GrupoDiagn_Neurologico', -1.2679312561932758), ('GrupoDiagn_Outro', -1.3336288354930728), ('GrupoDiagn_Respiratorio', -1.3122358932591094), ('Local_SU', -0.5473159680709174), ('Idade', 2.7632953690367095), ('Interna_Dias', 0.5604063368446363), ('SIRS', 0.26800101260763515), ('Glicose', 0.6610736507046577), ('Sodio', -0.18367756243089725), ('Ureia', 0.6048300074846206), ('Creatinina', 0.3561660588893121), ('PCR', 0.30114104140838155), ('pH', -0.3383519705679988), ('Ca_ionizado', -0.19972849199950723), ('pCO2', 0.564723049184796), ('pO2', -1.8445670878455789), ('HCO3', -0.6653350316911912), ('Genero', -0.9191410658874702), ('Antidislipidemicos', -0.3034352050351555), ('Antipsicoticos', 1.089770508615296), ('Antidepressores', 0.09359415285472611), ('Analgesicos', -0.7783542055065215), ('Anticoagulantes', 0.28705920388412215), ('Digitalicos', -0.14011655675577242), ('Corticosteroides', 0.22989736966962487), ('Outros Med_Presente', 1.5893706794527664), ('Alcoolico', 0.2503520395369789)]

def takeSecond(elem):
    return elem[1]


coefs.sort(key=takeSecond, reverse=True)

print("Labels")
for i in coefs:
	print(i[0], '\n')

print("########\n")
print("Values\n")
for i in coefs:
	print(i[1], '\n')
