from pathlib import Path

dataset_paths = {
	'celeba_train': Path(''),
	'celeba_test': Path(''),

	'ffhq': Path(''),
	'ffhq_unaligned': Path('')
}

model_paths = {
	# models for backbones and losses
	'ir_se50': Path('pretrained_models/model_ir_se50.pth'),
	# stylegan3 generators
	'stylegan3_ffhq': Path('pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'),
	'stylegan3_ffhq_pt': Path('pretrained_models/sg3-r-ffhq-1024.pt'),
	'stylegan3_ffhq_unaligned': Path('pretrained_models/stylegan3-r-ffhqu-1024x1024.pkl'),
	'stylegan3_ffhq_unaligned_pt': Path('pretrained_models/sg3-r-ffhqu-1024.pt'),
	# model for face alignment
	'shape_predictor': Path('pretrained_models/shape_predictor_68_face_landmarks.dat'),
	# models for ID similarity computation
	'curricular_face': Path('pretrained_models/CurricularFace_Backbone.pth'),
	'mtcnn_pnet': Path('pretrained_models/mtcnn/pnet.npy'),
	'mtcnn_rnet': Path('pretrained_models/mtcnn/rnet.npy'),
	'mtcnn_onet': Path('pretrained_models/mtcnn/onet.npy'),
	# classifiers used for interfacegan training
	'age_estimator': Path('pretrained_models/dex_age_classifier.pth'),
	'pose_estimator': Path('pretrained_models/hopenet_robust_alpha1.pkl')
}

styleclip_directions = {
	"ffhq": {
		'delta_i_c': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy'),
		's_statistics': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats'),
	},
	'templates': Path('editing/styleclip_global_directions/templates.txt')
}

interfacegan_aligned_edit_paths = {
	'5_0_Clock_Shadow': Path('editing/interfacegan/boundaries/ffhq/5_0_Clock_Shadow_boundary.npy'),
	'Age': Path('editing/interfacegan/boundaries/ffhq/age_boundary.npy'),
	'Arched_Eyebrows': Path('editing/interfacegan/boundaries/ffhq/Arched_Eyebrows_boundary.npy'),
	'Attractive': Path('editing/interfacegan/boundaries/ffhq/Attractive_boundary.npy'),
	'Bags_Under_Eyes': Path('editing/interfacegan/boundaries/ffhq/Bags_Under_Eyes_boundary.npy'),
	'Bald': Path('editing/interfacegan/boundaries/ffhq/Bald_boundary.npy'),
	'Bangs': Path('editing/interfacegan/boundaries/ffhq/Bangs_boundary.npy'),
	'Big_Lips': Path('editing/interfacegan/boundaries/ffhq/Big_Lips_boundary.npy'),
	'Big_Nose': Path('editing/interfacegan/boundaries/ffhq/Big_Nose_boundary.npy'),
	'Black_Hair': Path('editing/interfacegan/boundaries/ffhq/Black_Hair_boundary.npy'),
	'Blond_Hair': Path('editing/interfacegan/boundaries/ffhq/Blond_Hair_boundary.npy'),
	'Blurry': Path('editing/interfacegan/boundaries/ffhq/Blurry_boundary.npy'),
	'Brown_Hair': Path('editing/interfacegan/boundaries/ffhq/Brown_Hair_boundary.npy'),
	'Bushy_Eyebrows': Path('editing/interfacegan/boundaries/ffhq/Bushy_Eyebrows_boundary.npy'),
	'Chubby': Path('editing/interfacegan/boundaries/ffhq/Chubby_boundary.npy'),
	'Double_Chin': Path('editing/interfacegan/boundaries/ffhq/Double_Chin_boundary.npy'),
	'Eyeglasses': Path('editing/interfacegan/boundaries/ffhq/Eyeglasses_boundary.npy'),
	'Goatee': Path('editing/interfacegan/boundaries/ffhq/Goatee_boundary.npy'),
	'Gray_Hair': Path('editing/interfacegan/boundaries/ffhq/Gray_Hair_boundary.npy'),
	'Heavy_Makeup': Path('editing/interfacegan/boundaries/ffhq/Heavy_Makeup_boundary.npy'),
	'High_Cheekbones': Path('editing/interfacegan/boundaries/ffhq/High_Cheekbones_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhq/Male_boundary.npy'),
	'Mouth_Slightly_Open': Path('editing/interfacegan/boundaries/ffhq/Mouth_Slightly_Open_boundary.npy'),
	'Mustache': Path('editing/interfacegan/boundaries/ffhq/Mustache_boundary.npy'),
	'Narrow_Eyes': Path('editing/interfacegan/boundaries/ffhq/Narrow_Eyes_boundary.npy'),
	'No_Beard': Path('editing/interfacegan/boundaries/ffhq/No_Beard_boundary.npy'),
	'Oval_Face': Path('editing/interfacegan/boundaries/ffhq/Oval_Face_boundary.npy'),
	'Pale_Skin': Path('editing/interfacegan/boundaries/ffhq/Pale_Skin_boundary.npy'),
	'Pointy_Nose': Path('editing/interfacegan/boundaries/ffhq/Pointy_Nose_boundary.npy'),
	'Pose': Path('editing/interfacegan/boundaries/ffhq/pose_boundary.npy'),
	'Receding_Hairline': Path('editing/interfacegan/boundaries/ffhq/Receding_Hairline_boundary.npy'),
	'Rosy_Cheeks': Path('editing/interfacegan/boundaries/ffhq/Rosy_Cheeks_boundary.npy'),
	'Sideburns': Path('editing/interfacegan/boundaries/ffhq/Sideburns_boundary.npy'),
	'Smiling': Path('editing/interfacegan/boundaries/ffhq/Smiling_boundary.npy'),
	'Straight_Hair': Path('editing/interfacegan/boundaries/ffhq/Straight_Hair_boundary.npy'),
	'Wavy_Hair': Path('editing/interfacegan/boundaries/ffhq/Wavy_Hair_boundary.npy'),
	'Wearing_Earrings': Path('editing/interfacegan/boundaries/ffhq/Wearing_Earrings_boundary.npy'),
	'Wearing_Hat': Path('editing/interfacegan/boundaries/ffhq/Wearing_Hat_boundary.npy'),
	'Wearing_Lipstick': Path('editing/interfacegan/boundaries/ffhq/Wearing_Lipstick_boundary.npy'),
	'Wearing_Necklace': Path('editing/interfacegan/boundaries/ffhq/Wearing_Necklace_boundary.npy'),
	'Wearing_Necktie': Path('editing/interfacegan/boundaries/ffhq/Wearing_Necktie_boundary.npy'),
	'Young': Path('editing/interfacegan/boundaries/ffhq/Young_boundary.npy')
}

interfacegan_unaligned_edit_paths = {
	'age': Path('editing/interfacegan/boundaries/ffhqu/age_boundary.npy'),
	'smile': Path('editing/interfacegan/boundaries/ffhqu/Smiling_boundary.npy'),
	'pose': Path('editing/interfacegan/boundaries/ffhqu/pose_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhqu/Male_boundary.npy'),
}
