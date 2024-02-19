import pandas as pd

# Assuming df contains the loaded DataFrame with the column "NARRATIVE"
df = pd.read_excel("./SampleDataImages.xlsx", sheet_name="ImageDataCollection", usecols=["NARRATIVE"])

def segment_multiline_data(paragraph):
    # rows = []
    # lines = paragraph.split('\n')
    # modified_lines_array = [i for i in lines if not ('' == i)]
    # for line in modified_lines_array:
    #     rows.append({'Original Sentence': paragraph, 'NARRATIVE': line.strip()})  
    # return rows
    if isinstance(paragraph, str):
        rows = []
        lines = paragraph.split('\n')
        modified_lines_array = [i for i in lines if not ('' == i)]
        for line in modified_lines_array:
            rows.append({'Original Sentence': paragraph, 'NARRATIVE': line.strip()})  
        rows.extend([{'Original Sentence': '', 'NARRATIVE': ''}] * 2)
        return rows
    else:
        return []

segmented_rows = []
for index, row in df.iterrows():
    segmented_rows.extend(segment_multiline_data(row['NARRATIVE']))

segmented_df = pd.DataFrame(segmented_rows)

segmented_df.to_csv('segmented_data.csv', index=False)
