import typing
import numpy as np
import oneflow as flow
import matplotlib.pyplot as plt


@flow.global_function(type="predict")
def test_job() -> typing.Tuple[flow.typing.Numpy, flow.typing.Numpy]:
    batch_size = 64
    color_space = "RGB"

    with flow.scope.placement("cpu", "0:0"):
        # data/tbd00/part-00000 https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000
        ofrecord = flow.data.ofrecord_reader(('data/tbd00'), batch_size=batch_size,
                data_part_num=1, part_name_suffix_length=5, random_shuffle=True, shuffle_after_epoch=True)
        image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded", color_space=color_space)
        label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
        rsz = flow.image.Resize(image, resize_x=224, resize_y=224, color_space=color_space)
        rng = flow.random.CoinFlip(batch_size=batch_size)
        normal = flow.image.CropMirrorNormalize(rsz, mirror_blob=rng, color_space=color_space,
                mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], output_dtype=flow.float)
    return normal, label


if __name__ == "__main__":
    images, labels = test_job()
    print(images.shape, labels.shape)

    np_std = np.array([58.393, 57.12, 57.375])
    np_mean = np.array([123.68, 116.779, 103.939])
    z0 = np.round(images[0]*np_std.reshape(-1,1,1) + np_mean.reshape(-1,1,1)).astype(np.uint8)
    plt.imshow(z0.transpose(1,2,0))
