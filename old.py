import pandas as pd
import ipdb
import re
import bisect
from Levenshtein import distance
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

merged = pd.read_csv("merged.csv")
merged.drop(columns={'Date', 'Subcategory', 'CompText'}, inplace=True)
merged.rename(columns={'Snippet':'Fallacy', 'Fallacy':'Label', 'CompLabel':'arg_comp', 'RelLAbel':'arg_rel'}, inplace=True)

merged['Dialogue'] = merged['Dialogue'].apply(lambda x: re.sub(r'([a-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
merged['Dialogue'] = merged['Dialogue'].apply(lambda x: re.sub(r'([a-z]+\.)([A-Za-z]+)', r'\1 \2', x))
merged['Dialogue'] = merged['Dialogue'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
merged['Dialogue'] = merged['Dialogue'].apply(lambda x: x.replace("\n", ""))
merged['Fallacy'] = merged['Fallacy'].apply(lambda x: re.sub(r'([a-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
merged['Fallacy'] = merged['Fallacy'].apply(lambda x: re.sub(r'([a-z]+\.)([A-Za-z]+)', r'\1 \2', x))
merged['Fallacy'] = merged['Fallacy'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
merged['Fallacy'] = merged['Fallacy'].apply(lambda x: x.replace("\n", ""))

def get_context(fall, dial, fallacy_label, file_name, arg_comp, arg_rel):
    """ returns 
            the context of the fallacy (list of four elements):
            1) sentence before the sentence where the fallacy is located otherwise ''
            2) sentence where the fallacy is located
            3) sentence after the sentence where the fallacy is located otherwhise ''
            4) the fallacy snippet
            the sentences out of context (not including the ones of context)
    """
    ctx = []
    dial_sents = sent_tokenize(dial)
    for i in range(len(dial_sents)):
        falls = sent_tokenize(fall)
        len_falls = len(falls)
        if len_falls == 1:
            #if fall in dial_sents[i]:
            if re.search(r'\b{}\b'.format(fall if fall[-1]!='.' else fall[:-1]), dial_sents[i]):
                ctx.append(file_name[:4])
                ctx.append(dial_sents[i-1] if i>=1 else '')
                ctx.append(dial_sents[i])
                ctx.append(dial_sents[i+1]if i+2 <= len(dial_sents) else '')
                ctx.append(fall)
                ctx.append(fallacy_label)
                ctx.append(str(arg_comp))
                ctx.append(str(arg_rel))
                break
        else:
        #if falls[0] in dial_sents[i]:
            if re.search(r'\b{}\b'.format(falls[0] if falls[0][-1]!='.' else falls[0][:-1]), dial_sents[i]):
                ctx.append(file_name[:4])
                ctx.append(dial_sents[i-1] if i>=1 else '')
                curr = ''
                for x in falls:
                    curr = curr + " " + x
                ctx.append(curr)
                ctx.append(dial_sents[i+len_falls] if i+2+len_falls <= len(dial_sents) else '')
                ctx.append(fall)
                ctx.append(fallacy_label)
                ctx.append(str(arg_comp))
                ctx.append(str(arg_rel))
                break

    #ooc list is for out of context sentences (out of 3 sentences from dialogue of fallacy)
    ctx_sent = sent_tokenize(' '.join(ctx))
    ooc = []
    for x in dial_sents:
        if x not in ctx_sent:
            ooc.append(x)
    return ctx, ooc

def tokenize_with_offsets(dial, fallacy):
    """ returns token, start and end offset of main text
        returns start and end offset of subtext considering the offsets of the main text
    """
    # ipdb.set_trace()
    d_tok, d_starts, d_ends = zip(*[(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', dial)])
    f_tok, f_starts, f_ends = zip(*[(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', fallacy)])

    print(f_tok, f_starts, f_ends)

    dial_dict = {}
    for t, s, e in zip(d_tok, d_starts, d_ends):
        dial_dict[(s,e)]=re.sub(r'[^\w\s]', '', t)

    d_list = list(d_tok)
    f_list = list(f_tok)
    print(d_list)
    print(f_list)
    ner = []
    #if is_Sublist(d_list, f_list):
    #if re.search(r'\b{}'.format(fallacy if fallacy[-1]!='.' else fallacy[:-1]), dial):
    for i in range(len(f_list)):
        if i == 0:
            #start = ([k for k, v in dial_dict.items() if re.sub(r'[^\w\s]', '', v) == re.sub(r'[^\w\s]', '', f_list[i])][0][0])
            start = ([k for k, v in dial_dict.items() if (distance(re.sub(r'[^\w\s]', '', f_list[i]), re.sub(r'[^\w\s]', '', v)) <= 3 )][0][0])
        if i == len(f_list)-1:
            print(i, f_list[i], f_list.index(f_list[i]))
            end = ([k for k, v in dial_dict.items() if re.sub(r'[^\w\s]', '', v) == re.sub(r'[^\w\s]', '', f_list[i])][0][1])        
            #x = [k for k, v in dial_dict.items() if (distance(re.sub(r'[^\w\s]', '', v), re.sub(r'[^\w\s]', '', f_list[i])) <= 3)]
            #end = ([k for k, v in dial_dict.items() if (distance(re.sub(r'[^\w\s]', '', v), re.sub(r'[^\w\s]', '', f_list[i])) <= 3)][0][1])        
    return d_tok, d_starts, d_ends, [(start, end)]

def get_labels(dial, fall, label):
    """ Convert offsets to sequence labels in BIO format."""
    d_t, d_s, d_e, span = tokenize_with_offsets(dial, fall)
    labels = ["O"]*len(d_s)
    spans = sorted(span)
    for s, e in spans:
        li = bisect.bisect_left(d_s, s)
        ri = bisect.bisect_left(d_s, e)
        ni = len(labels[li:ri])
        labels[li] = f"B-{label}"
        labels[li+1:ri] = [f"I-{label}"]*(ni-1)
    return labels

def operation(row):
    return get_labels(row['Context'], row['Fallacy'], row['Label'])

# all_ctx = []
# all_ooc = []
# x = 0
# for i, row in merged.iterrows():
#     fallacy = row['Fallacy']
#     context = row['Dialogue']
#     fallacy_label = row['Label']
#     ctx, ooc = get_context(fallacy, context, fallacy_label, row['FileName'], row['arg_comp'], row['arg_rel'])
#     if ctx:
#         all_ctx.append(ctx)
#     all_ooc.append(ooc)
  
# old_df = pd.DataFrame(all_ctx, columns = ['Date', 'PreviousSentence', 'CurrentSentence', 'NextSentence', 'Fallacy', 'Label', 'arg_comp', 'arg_rel'])
# old_df.drop_duplicates(inplace=True)
# old_df = old_df.reset_index(drop=True)
# old_df['Context'] = old_df['PreviousSentence'].astype(str) + old_df['CurrentSentence'].astype(str) + old_df['NextSentence'].astype(str)
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)(\[[A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)([0-9]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)(\([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.\")([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.\'\')([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([0-9]+\.)([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([0-9]+\%\.)([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
# old_df['Context'] = old_df['Context'].apply(lambda x: x.replace("\n", ""))
# old_df['Context'] = old_df['Context'].apply(lambda x: " ".join(x.split()))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: re.sub(r'([A-Za-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: re.sub(r'([A-Za-z]+\.)([A-Za-z]+)', r'\1 \2', x))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: re.sub(r'([A-Za-z]+\.\")([A-Za-z]+)', r'\1 \2', x))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: re.sub(r'([0-9]+\.)([A-Za-z]+)', r'\1 \2', x))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: re.sub(r'([0-9]+\%\.)([A-Za-z]+)', r'\1 \2', x))

# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: x.replace("\n", ""))
# old_df['Fallacy'] = old_df['Fallacy'].apply(lambda x: " ".join(x.split()))

# old_df.drop(columns={'PreviousSentence', 'CurrentSentence', 'NextSentence'},inplace=True)

# old_df['bio_tags'] = old_df.apply(operation, axis=1)


new = pd.read_csv("new.csv")
# old_df['Context'] = old_df['PreviousSentence'].astype(str) + old_df['CurrentSentence'].astype(str) + old_df['NextSentence'].astype(str)
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)(\[[A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)([0-9]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)(\([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.)([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.\")([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([A-Za-z]+\.\'\')([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([0-9]+\.)([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([0-9]+\%\.)([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: re.sub(r'([a-z]+\?)([A-Za-z]+)', r'\1 \2', x))
new['Context'] = new['Context'].apply(lambda x: x.replace("\n", ""))
new['Context'] = new['Context'].apply(lambda x: " ".join(x.split()))
new['Fallacy'] = new['Fallacy'].apply(lambda x: re.sub(r'([A-Za-z]+)(\-)([A-Za-z]+)', r'\1 \2 \3', str(x)))
new['Fallacy'] = new['Fallacy'].apply(lambda x: re.sub(r'([A-Za-z]+\.)([A-Za-z]+)', r'\1 \2', x))
new['Fallacy'] = new['Fallacy'].apply(lambda x: re.sub(r'([A-Za-z]+\.\")([A-Za-z]+)', r'\1 \2', x))
new['Fallacy'] = new['Fallacy'].apply(lambda x: re.sub(r'([0-9]+\.)([A-Za-z]+)', r'\1 \2', x))
new['Fallacy'] = new['Fallacy'].apply(lambda x: re.sub(r'([0-9]+\%\.)([A-Za-z]+)', r'\1 \2', x))
new['bio_tags'] = new.apply(operation, axis=1)

old_test = pd.read_csv("old_test_set.csv")
old_train = pd.read_csv("old_train_set.csv")

mix = [old_test, old_train, new]
final_gold = pd.concat(mix).reset_index(drop=True)

train_set, test_set = train_test_split(final_gold, stratify=final_gold['Label'], test_size=0.1, random_state=42)
train_set.to_csv("gold_train_set.csv", index=False)
test_set.to_csv("gold_test_set.csv", index=False)

ipdb.set_trace()
