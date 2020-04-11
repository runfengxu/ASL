
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import json

if __name__ == "__main__":

    label = "eyebrows"
    res_fold = "{}/".format(label) #res
    src_path = "781_12582.xml" #the pth for xml file
    tree = ET.parse(src_path)
    root = tree.getroot()

    c_list = root.findall("./COLLECTIONS/")#get collects

    valid_num = 0
    unvalid_num = 0

    signer_num = {}
    signer_valid = {}

    single_num = 0
    repeat_num = 0


    for c_ele in c_list:
        ann_info = c_ele.findall("./TEMPORAL-PARTITIONS/TEMPORAL-PARTITION/SEGMENT-TIERS/SEGMENT-TIER/")
        #print(len(ann_info))
        assert len(ann_info) == 3
        signer = ann_info[0].text
        utt_list = ann_info[2].findall("UTTERANCE")

        
        for utt in utt_list:
            utt_info = {}
            try:
                signer_num[signer] += 1
            except:
                signer_num[signer] = 1


            utt_id = utt.get("ID")
            u_s_frame = int(utt.get("START_FRAME"))
            u_s_frame_end = int(utt.get("END_FRAME"))
            non_manual_list = utt.findall("./NON_MANUALS/NON_MANUAL")



            utt_info["uid"] = utt_id
            utt_info["signer"] = signer
            utt_info["eyebrow_action"]=[]
            utt_info['length']= int((u_s_frame_end - u_s_frame)/1001)

            for ele in non_manual_list:
                label = ele.find("LABEL").text
                value = ele.find("VALUE").text
                onset = ele.find("ONSET")
                offset = ele.find("OFFSET")

                # print(label)
                # if (label == '\'eye brows\''):
                #     print(value)
                # print(utt_id)
                if label == '\'eye brows\'' and (value[1:-1] in {'further raised','raised','slightly raised','slightly lowered','lowered','further lowered'}):
                    # print('find')
                    s_frame = int((int(ele.get("START_FRAME"))-u_s_frame) / 1001)
                    e_frame = int((int(ele.get("END_FRAME"))-u_s_frame)/1001)
                    if  onset is not None:
                        onset = (int((int(onset.get("START_FRAME"))-u_s_frame)/1001),int((int(onset.get("END_FRAME"))-u_s_frame)/1001))
                    else: onset  = None
                    if offset is not None:
                        offset = (int((int(offset.get("START_FRAME"))-u_s_frame)/1001),int((int(offset.get("END_FRAME"))-u_s_frame)/1001))
                    else: offset = None

                    utt_info["eyebrow_action"].append({'action':value,'s':s_frame,'e':e_frame,'onset':onset,'offset':offset})
            if utt_info["eyebrow_action"]!=[]:
                res_path = res_fold + "{}.json".format(utt_id)
                res_file = open(res_path, "w")
                res_file.write(json.dumps(utt_info))
            print('finish:',utt_id)