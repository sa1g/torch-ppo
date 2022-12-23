dizionario = {
    '1': 'ciao',
    '2': 'pcrd'
}

cacca = {
    '2': 'daiods'
}

for (item, aa), abla in zip(dizionario.items(), cacca.values()):
    print(item)
    print(aa)
    print(abla)