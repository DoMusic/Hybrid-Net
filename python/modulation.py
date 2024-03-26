from collections import defaultdict
from itertools import product

from common import MAJOR_CHORDS, MINOR_CHORDS, MAJ2MIN_MAP


def get_chord_tonic(chord):
    chord = chord.replace('Dim', 'dim').replace('Aug', 'aug')
    if len(chord) >= 3:
        if chord[1] in ["#", 'b']:
            if 'maj7' in chord:
                return chord[:2]
            # elif 'dim' in chord:
            #     return chord[:2] + 'm'
            elif chord[2] in ['m']:
                return chord[:3]
            elif 'sus2' in chord or 'sus4' in chord or 'dim' in chord or 'aug' in chord:
                return chord
            else:
                return chord[:2]
        elif chord[1] == "5":
            return chord[:2]
        # elif 'dim' in chord:
        #     return chord[:1] + 'm'
        elif 'sus2' in chord or 'sus4' in chord or 'dim' in chord or 'aug' in chord:
            return chord
        elif 'maj7' in chord:
            return chord.replace('maj7', '')
        elif '7M' in chord:
            return chord.replace('7M', '')
        elif chord[1] in ['m']:
            return chord[:2]
        else:
            return chord[:1]
    elif len(chord) == 2:
        if chord[1] in ['#', 'b', 'm', '5']:
            return chord[:2]
        else:
            return chord[:1]
    else:
        return chord[:1]


def chord_name_filter(chords):
    search_name_list = []
    for i in chords:
        if not i:
            continue
        search_name_list.append(get_chord_tonic(i))
    return search_name_list


def chord_to_tone(chord):
    tone = chord[0]
    if len(chord) > 2:
        if chord[1] in ['b', '#']:
            tone += chord[1]
    return tone


def get_chords_repeat_num3(chords):
    chords_score_map = {}
    for ex_chords in MAJOR_CHORDS:
        chords_score_map[ex_chords[0]] = len([i for i in chords if i in ex_chords])
    return chords_score_map


def process_sus_chords(filter_total_chords, sus_chords):
    tone_count_dic = defaultdict(int)
    for chord in sus_chords:
        tone_count_dic[chord] += 1
    keys = list(tone_count_dic)
    product_list = list(product(*([[0, 1]] * len(keys))))
    chords_score_list = []
    for item in product_list:
        sus_conv_chord = []
        for i, x in enumerate(item):
            key = keys[i]
            tone = chord_to_tone(key)
            if x == 0:
                sus_conv_chord.extend([tone] * tone_count_dic[key])
            else:
                sus_conv_chord.extend([tone + 'm'] * tone_count_dic[key])
        chords_score_map = get_chords_repeat_num3(filter_total_chords + sus_conv_chord)
        chords_score_list.extend([(k, i, item) for k, i in chords_score_map.items() if i != 0])
    chords_score_list = sorted(chords_score_list, key=lambda x: x[1], reverse=True)
    return chords_score_list, keys


def get_ret_key(best_key, chord, keys, maj_min_dic, is_sus, ret_key, sus_dic):
    # print(best_key, "best_key", chord)
    for k, i, item in best_key:
        if 'sus' in chord or 'dim' in chord or 'aug' in chord or '5' in chord:
            tone = chord_to_tone(chord)
            if item[keys.index(chord)] == 1:
                tone += 'm'
            chord = tone
        if chord == maj_min_dic[k][0] or chord == maj_min_dic[k][5]:
            ret_key = k
            if is_sus:
                sus_dic = {keys[i]: chord_to_tone(keys[i]) if v == 0 else chord_to_tone(keys[i]) + 'm' for i, v in
                           enumerate(item)}
            break
    return ret_key, sus_dic


def get_ret_key2(best_key, head_chord, tail_chord, maj_min_dic, ret_key):
    for k, i in best_key:
        if head_chord == maj_min_dic[k][0] or head_chord == maj_min_dic[k][5]:
            ret_key = k
            break
    if not ret_key:
        for k, i in best_key:
            if tail_chord == maj_min_dic[k][0] or tail_chord == maj_min_dic[k][5]:
                ret_key = k
                break
    return ret_key


def process_best_key(chords_score_list, head_chord, tail_chord, keys, maj_min_dic, ret_key, sus_dic, is_sus=True):
    # 获取key
    if not chords_score_list:
        return None, None
    if is_sus:
        best_key = [x for x in chords_score_list if x[1] == chords_score_list[0][1]]
    else:
        best_key = [x for x in chords_score_list if x[-1] == chords_score_list[0][-1]]
    if len(best_key) > 1 and is_sus:
        ret_key, sus_dic = get_ret_key(best_key, head_chord, keys, maj_min_dic, is_sus, ret_key, sus_dic)
        if not ret_key:
            ret_key, sus_dic = get_ret_key(best_key, tail_chord, keys, maj_min_dic, is_sus, ret_key, sus_dic)
    elif len(best_key) > 1 and not is_sus:
        ret_key = get_ret_key2(best_key, head_chord, tail_chord, maj_min_dic, ret_key)
    # else:
    #     return None, None
    if not ret_key:
        if len(best_key[0]) == 2:
            ret_key = best_key[0][0]
        else:
            ret_key = best_key[0][0]
            sus_dic = {keys[i]: chord_to_tone(keys[i]) if v == 0 else chord_to_tone(keys[i]) + 'm' for i, v in
                       enumerate(best_key[0][2])}
    return ret_key, sus_dic


def get_maj_or_min_key(ret_key, maj_min_dic, head_chord, tail_chord, sus_dic):
    # check maj or min
    maj_key = ret_key
    min_key = MAJ2MIN_MAP[ret_key]

    maj_chords = maj_min_dic[maj_key]

    if 'sus' in head_chord or 'dim' in head_chord or 'aug' in head_chord or '5' in head_chord:
        head_chord = sus_dic[head_chord]

    if head_chord == maj_chords[0]:
        ret_key = maj_key
    elif head_chord == maj_chords[5]:
        ret_key = min_key
    else:
        if 'sus' in tail_chord or 'dim' in tail_chord or 'aug' in tail_chord or '5' in tail_chord:
            tail_chord = sus_dic[tail_chord]

        if tail_chord == maj_chords[0]:
            ret_key = maj_key
        elif tail_chord == maj_chords[5]:
            ret_key = min_key
        else:
            ret_key = maj_key
    return ret_key


def search_key(total_chords):
    if not total_chords:
        return "C", 0
    maj_min_dic = {x[0]: x for x in MAJOR_CHORDS + MINOR_CHORDS}
    ret_key = None
    keys = []
    sus_dic = {}
    total_chords = chord_name_filter(total_chords)
    head_chord = total_chords[0]
    tail_chord = total_chords[-1]

    sus_chords = [x for x in total_chords if 'sus' in x or 'dim' in x or 'aug' in x or '5' in x]
    filter_total_chords = [x for x in total_chords if
                           'sus' not in x and 'dim' not in x and 'aug' not in x and '5' not in x]
    if not filter_total_chords and not sus_chords:
        return "C", 0

    if sus_chords:
        chords_score_list, keys = process_sus_chords(filter_total_chords, sus_chords)

        ret_key, sus_dic = process_best_key(chords_score_list, head_chord, tail_chord, keys, maj_min_dic, ret_key,
                                            sus_dic)
    else:
        chords_score_map = get_chords_repeat_num3(filter_total_chords)
        chords_score_list = [(k, i) for k, i in chords_score_map.items() if i != 0]
        chords_score_list = sorted(chords_score_list, key=lambda x: x[1], reverse=True)
        print(chords_score_list)
        ret_key, sus_dic = process_best_key(chords_score_list, head_chord, tail_chord, keys, maj_min_dic, ret_key,
                                            sus_dic, is_sus=False)
    key_score = 0
    for i in chords_score_list:
        if i[0] == ret_key:
            key_score = i[1]
    if not ret_key:
        return "C", 0
    ret_key = get_maj_or_min_key(ret_key, maj_min_dic, head_chord, tail_chord, sus_dic)

    return ret_key, key_score


def chord_convert(chord):
    chord_convert_map = {
        "C": "Db",
        "D": "Eb",
        "E": "F",
        "F": "Gb",
        "G": "Ab",
        "A": "Bb",
        "B": "C"
    }
    if "#" in chord:
        a, b = chord.split('#')
        new_chord = chord_convert_map[a] + b
    else:
        new_chord = chord
    return new_chord


if __name__ == '__main__':
    demo_chords = ['D', 'D', 'Bm7', 'Bm7', 'Asus4', 'A', 'G', 'G', 'D', 'D', 'D', 'Bm7', 'A', 'Asus4', 'A', 'G', 'D',
                   'D', 'D', 'D', 'Bm7', 'A', 'Asus4', 'A', 'G', 'D', 'E', 'D', 'D', 'Bm', 'Bm7', 'A', 'A', 'G', 'G',
                   'D', 'D', 'Bm7', 'Bm', 'Bm7', 'A', 'A', 'D', 'G', 'G', 'Bm7', 'Bm', 'A', 'G', 'G', 'A', 'D', 'D',
                   'D', 'D', 'A', 'Asus4', 'A', 'Asus4', 'Em7', 'G', 'D', 'D', 'D', 'D', 'D', 'A', 'A7', 'A7', 'G', 'G',
                   'D', 'D', 'Bm7', 'Bm7', 'A', 'A', 'G', 'G', 'D', 'D', 'Bm', 'Bm7', 'A', 'Asus4', 'A', 'D', 'G', 'G',
                   'Bm', 'A', 'G', 'G', 'A', 'G', 'G', 'Abm', 'Dbm', 'B', 'A', 'Abm7', 'Ab7', 'A', 'Gbm7', 'E', 'Gbm7',
                   'Ab7', 'Dbm7', 'B', 'A', 'E', 'Ab7', 'A', 'E', 'Gbm7', 'Ab7', 'Dbm', 'B', 'A', 'A', 'B', 'E', 'B',
                   'Abm', 'A', 'E', 'B', 'Abm7', 'Abm', 'A', 'F', 'C', 'A', 'Bb', 'F', 'C', 'A', 'F', 'F', 'Bb', 'Bb',
                   'F', 'F', 'Dm7', 'D', 'Dm7', 'C', 'C', 'Bb', 'Bb', 'F', 'F', 'C', 'Dm7', 'D', 'Dm7', 'F', 'Gsus4',
                   'C', 'Bb', 'Bb', 'Bb', 'Bb', 'F', 'F']
    demo_chords1 = []
    for i in demo_chords:
        demo_chords1.append(chord_convert(i))
    print(search_key(demo_chords))
    print(search_key(demo_chords1))
