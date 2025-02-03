import csv

input_file = '/Users/gabrielchavez/Documents/Farinter/ia/Documentos/DL_Kielsa_empleadosxhora.csv'
output_file = 'DL_Kielsa_empleadosxhora_no_quotes.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Eliminar comillas dobles de cada campo
        row = [field.replace('"', '') for field in row]
        writer.writerow(row)

print(f'Comillas dobles eliminadas y guardadas en {output_file}')