import io
import os
import PIL.Image
import numpy as np
import mindspore as ms
import mindspore.mindrecord

from utils import next_tbd_dir


def demo_simple_mindrecord():
    logdir = next_tbd_dir()
    mindrecord_file = os.path.join(logdir, 'test.mindrecord')
    # test.mindrecord, test.mindrecord.db

    cv_schema = {'filename':{'type':'string'}, 'label':{'type':'int32'}, 'data':{'type':'bytes'}}
    writer = ms.mindrecord.FileWriter(mindrecord_file, shard_num=1)
    writer.add_schema(cv_schema, 'a cv dataset')
    writer.add_index(["filename", "label"])

    N0 = 23
    height_width = np.random.randint(28, 40, size=(N0,2))
    np_image = [np.random.randint(0,255,size=(x,y,3),dtype=np.uint8) for x,y in height_width]
    np_label = np.random.randint(0, 10, size=(N0,)).astype(np.int32) #dtype must be the SAME as the schema
    filename_list = [f'image{x}.jpg' for x in range(N0)]

    for image,label,filename in zip(np_image, np_label, filename_list):
        tmp0 = io.BytesIO()
        PIL.Image.fromarray(image).save(tmp0, 'PNG') #JPEG is data-loss
        item = {'filename':filename, 'label':label, 'data':tmp0.getvalue()}
        writer.write_raw_data([item]) #one could include multiple item
    writer.commit()

    decode_op = ms.dataset.vision.c_transforms.Decode()
    ds0 = ms.dataset.MindDataset(dataset_file=mindrecord_file, shuffle=False)
    ds0 = ds0.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
    ds_iter = ds0.create_dict_iterator(output_numpy=True)
    tmp0 = list(ds_iter)
    image_mr = [x['data'] for x in tmp0]
    label_mr = [x['label'] for x in tmp0]
    filename_mr = [x['filename'] for x in tmp0]

    assert all(x.shape==y.shape for x,y in zip(np_image,image_mr))
    assert all(np.all(x==y) for x,y in zip(np_image,image_mr)) #TODO strange
    assert all(x==y for x,y in zip(np_label,label_mr))
    assert all(x==y for x,y in zip(np_label,label_mr))
