import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import dlib
import subprocess

from utils.alignment_utils import align_face, crop_face, get_stylegan_transform


ENCODER_PATHS = {
    "restyle_e4e_ffhq": {"id": "1z_cB187QOc6aqVBdLvYvBjoc93-_EuRm", "name": "restyle_e4e_ffhq.pt"},
    "restyle_pSp_ffhq": {"id": "12WZi2a9ORVg-j6d9x4eF-CKpLaURC2W-", "name": "restyle_pSp_ffhq.pt"},
}
INTERFACEGAN_PATHS = {
    "5_0_Clock_Shadow": {'id': '1uur6Ro3bbaVsimTdWhVhcmHzQUDlBSCh', 'name': '5_0_Clock_Shadow_boundary.npy'},
    "Age": {'id': '1NQVOpKX6YZKVbz99sg94HiziLXHMUbFS', 'name': 'age_boundary.npy'},
    "Arched_Eyebrows": {'id': '1-ib5htTg1f8r9fFa5LuGq7jPpQWhDn2h', 'name': 'Arched_Eyebrows_boundary.npy'},
    "Attractive": {'id': '1gjWcWuAs3oktqiup47a6wJZXdDO30ekc', 'name': 'Attractive_boundary.npy'},
    "Bags_Under_Eyes": {'id': '1AsYP-yaClhwfRbw5q2eunrU0cj-An1o7', 'name': 'Bags_Under_Eyes_boundary.npy'},
    "Bald": {'id': '1T5IOdRC_1HF7ZTSidxcIxTiFnOsmTRbB', 'name': 'Bald_boundary.npy'},
    "Bangs": {'id': '1cD_UDzmuv0dpEjEneh_YzibHhyth68tV', 'name': 'Bangs_boundary.npy'},
    "Big_Lips": {'id': '1QCAVpzdsiZo-DV4aBaaTWdG1_2vypepo', 'name': 'Big_Lips_boundary.npy'},
    "Big_Nose": {'id': '1is7BUj5D6FCSi2iqRLudRyHFQwjgW3g6', 'name': 'Big_Nose_boundary.npy'},
    "Black_Hair": {'id': '1RjR-RG_3UU6jF9PSy5v1ioLDD5qUidKp', 'name': 'Black_Hair_boundary.npy'},
    "Blond_Hair": {'id': '16lXe-3I-cfPisSu-9sPqJrFRIduIBDPg', 'name': 'Blond_Hair_boundary.npy'},
    "Blurry": {'id': '16Qv-IG9Rkfzk7UdkRGRUuZozCyTMe1XI', 'name': 'Blurry_boundary.npy'},
    "Brown_Hair": {'id': '1Kr5t6MT6NW6CCFGC4jfrTSGICi3UAYwF', 'name': 'Brown_Hair_boundary.npy'},
    "Bushy_Eyebrows": {'id': '1xn0PK6cPlGyAewCv3wwjJfKGijratT0m', 'name': 'Bushy_Eyebrows_boundary.npy'},
    "Chubby": {'id': '1FwpsDY4iqRjmhBCdaGtsbP4sKjhxueZL', 'name': 'Chubby_boundary.npy'},
    "Double_Chin": {'id': '1qynI23Hwe_vxa9KCrApldJR-n7JIydqC', 'name': 'Double_Chin_boundary.npy'},
    "Eyeglasses": {'id': '1PCIq-nhlIgfLwmWetY7gEbbAIUos3gsN', 'name': 'Eyeglasses_boundary.npy'},
    "Goatee": {'id': '1rerLVAJkeEj-VqMycw4gI0IxPFgKfHg9', 'name': 'Goatee_boundary.npy'},
    "Gray_Hair": {'id': '1bKblTxzGEFeSbit-9GmhNctWHDVdjS8H', 'name': 'Gray_Hair_boundary.npy'},
    "Heavy_Makeup": {'id': '11M6k3e0OiBeta6Uoaq9B6ZLUMYIS0OfT', 'name': 'Heavy_Makeup_boundary.npy'},
    "High_Cheekbones": {'id': '1PEbv_ZJlFmX024M2LUw-CizyYpx0RhNP', 'name': 'High_Cheekbones_boundary.npy'},
    "Male": {'id': '18dpXS5j1h54Y3ah5HaUpT03y58Ze2YEY', 'name': 'Male_boundary.npy'},
    "Mouth_Slightly_Open": {'id': '1CtCS9EJZzrbXPSxpr4s5gL2UsbwxZC51', 'name': 'Mouth_Slightly_Open_boundary.npy'},
    "Mustache": {'id': '1uLzxtjW25kz8M0r6SBrHSxr6sE_DLHz5', 'name': 'Mustache_boundary.npy'},
    "Narrow_Eyes": {'id': '1fuZ6xiCL4v-Pzk9ry_HPV-FcDdrtf2Nj', 'name': 'Narrow_Eyes_boundary.npy'},
    "No_Beard": {'id': '1Q9hyBPmfM7Qa8grCkep4r0_IKamNCcS_', 'name': 'No_Beard_boundary.npy'},
    "Oval_Face": {'id': '1Q_BuZS42b81hbf2xKO1syI72ZLzk4RYf', 'name': 'Oval_Face_boundary.npy'},
    "Pale_Skin": {'id': '169W9aYvoxSAxinA3calJ6vvw5F4DJywa', 'name': 'Pale_Skin_boundary.npy'},
    "Pointy_Nose": {'id': '12GDUBgfsBpj4Dxw_whEbKOraICOhd6Ew', 'name': 'Pointy_Nose_boundary.npy'},
    "Pose": {'id': '1nCzCR17uaMFhAjcg6kFyKnCCxAKOCT2d', 'name': 'pose_boundary.npy'},
    "Receding_Hairline": {'id': '14_kHs6oy5AeNZkohw6s6h2eck5xjfypW', 'name': 'Receding_Hairline_boundary.npy'},
    "Rosy_Cheeks": {'id': '1fac4O3RT9BLgTbeNaogeHZZNpdnGAcon', 'name': 'Rosy_Cheeks_boundary.npy'},
    "Sideburns": {'id': '136JXtLi2XfbTJQEPh1eeedzSxh-BPbDA', 'name': 'Sideburns_boundary.npy'},
    "Smiling": {'id': '1KgfJleIjrKDgdBTN4vAz0XlgSaa9I99R', 'name': 'Smiling_boundary.npy'},
    "Straight_Hair": {'id': '13y_TuurStYG35cGEldOPioMKyoEIp9uT', 'name': 'Straight_Hair_boundary.npy'},
    "Wavy_Hair": {'id': '1jCNxtfHb1oaR63AqlmRHmUpPfN-UZ1dn', 'name': 'Wavy_Hair_boundary.npy'},
    "Wearing_Earrings": {'id': '1UwO4ukm5D6AD89SeKz4w1Wdp3ginTL_s', 'name': 'Wearing_Earrings_boundary.npy'},
    "Wearing_Hat": {'id': '1Bi5YHMzwYCur4R0WPBNc35Bxcyo6cFtP', 'name': 'Wearing_Hat_boundary.npy'},
    "Wearing_Lipstick": {'id': '15NfovFTs3RBF5G7IYukR4hIPAdZ1NEee', 'name': 'Wearing_Lipstick_boundary.npy'},
    "Wearing_Necklace": {'id': '1GoidOO2wZqZ8ln-LpyWxlz-m53K0N-5Z', 'name': 'Wearing_Necklace_boundary.npy'},
    "Wearing_Necktie": {'id': '1zqBvxyYDKEGii-1ez33Oy2aAjMI3v-qv', 'name': 'Wearing_Necktie_boundary.npy'},
    "Young": {'id': '1B-a7EJfFRQTvC7WWpqVODu2QEhyIl2c0', 'name': 'Young_boundary.npy'}
}
STYLECLIP_PATHS = {
    "delta_i_c": {"id": "1HOUGvtumLFwjbwOZrTbIloAwBBzs2NBN", "name": "delta_i_c.npy"},
    "s_stats": {"id": "1FVm_Eh7qmlykpnSBN1Iy533e_A2xM78z", "name": "s_stats"},
}


class Downloader:

    def __init__(self, code_dir, use_pydrive, subdir):
        self.use_pydrive = use_pydrive
        current_directory = os.getcwd()
        self.save_dir = os.path.join(os.path.dirname(current_directory), code_dir, subdir)
        os.makedirs(self.save_dir, exist_ok=True)
        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, file_id, file_name):
        file_dst = f'{self.save_dir}/{file_name}'
        if os.path.exists(file_dst):
            print(f'{file_name} already exists!')
            return
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id': file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            command = self._get_download_model_command(file_id=file_id, file_name=file_name)
            subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    def _get_download_model_command(self, file_id, file_name):
        """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=self.save_dir)
        return url


def download_dlib_models():
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')


def run_alignment(image_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Aligning image...")
    aligned_image = align_face(filepath=str(image_path), detector=detector, predictor=predictor)
    print(f"Finished aligning image: {image_path}")
    return aligned_image


def crop_image(image_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Cropping image...")
    cropped_image = crop_face(filepath=str(image_path), detector=detector, predictor=predictor)
    print(f"Finished cropping image: {image_path}")
    return cropped_image


def compute_transforms(aligned_path, cropped_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Computing landmarks-based transforms...")
    res = get_stylegan_transform(str(cropped_path), str(aligned_path), detector, predictor)
    print("Done!")
    if res is None:
        print(f"Failed computing transforms on: {cropped_path}")
        return
    else:
        rotation_angle, translation, transform, inverse_transform = res
        return inverse_transform
