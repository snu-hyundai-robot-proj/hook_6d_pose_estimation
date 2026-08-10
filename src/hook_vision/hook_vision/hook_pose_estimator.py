import os, time, math, itertools, traceback, cv2, torch
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from rfdetr import RFDETRSeg2XLarge

import zivid
import rclpy
from rclpy.node import Node


class CadCandidate:
    def __init__(self, name, transformation, coarse_fitness, coarse_rmse_mm,
                 trimmed_rmse_mm, median_mm, p90_mm, p95_mm,
                 ir1, ir2, ir4, normal_median_deg, quality_pass):
        self.name = name
        self.transformation = transformation
        self.coarse_fitness = coarse_fitness
        self.coarse_rmse_mm = coarse_rmse_mm
        self.trimmed_rmse_mm = trimmed_rmse_mm
        self.median_mm = median_mm
        self.p90_mm = p90_mm
        self.p95_mm = p95_mm
        self.ir1 = ir1
        self.ir2 = ir2
        self.ir4 = ir4
        self.normal_median_deg = normal_median_deg
        self.quality_pass = quality_pass


class CanonicalCandidate:
    def __init__(self, name, transformation,
                 stable_median_mm, stable_p90_mm, stable_p95_mm,
                 stable_ir1, stable_ir2, stable_ir3,
                 core_median_mm, core_p90_mm, core_p95_mm, core_ir2,
                 cad_median_mm, cad_p90_mm, cad_p95_mm, cad_ir2,
                 correction_translation_mm, correction_rotation_deg,
                 trimmed_rmse_mm, quality_pass):
        self.name = name
        self.transformation = transformation
        self.stable_median_mm = stable_median_mm
        self.stable_p90_mm = stable_p90_mm
        self.stable_p95_mm = stable_p95_mm
        self.stable_ir1 = stable_ir1
        self.stable_ir2 = stable_ir2
        self.stable_ir3 = stable_ir3
        self.core_median_mm = core_median_mm
        self.core_p90_mm = core_p90_mm
        self.core_p95_mm = core_p95_mm
        self.core_ir2 = core_ir2
        self.cad_median_mm = cad_median_mm
        self.cad_p90_mm = cad_p90_mm
        self.cad_p95_mm = cad_p95_mm
        self.cad_ir2 = cad_ir2
        self.correction_translation_mm = correction_translation_mm
        self.correction_rotation_deg = correction_rotation_deg
        self.trimmed_rmse_mm = trimmed_rmse_mm
        self.quality_pass = quality_pass


class HookPoseEstimator(Node):
    def __init__(self):
        super().__init__('hook_pose_estimator_node')

        self.declare_parameter('hand_side', 'Left')
        self.hand_side = self.get_parameter('hand_side').value

        if self.hand_side.lower() not in ['left', 'right']:
            raise ValueError("hand_side must be 'Left' or 'Right'")

        home_dir = os.path.expanduser('~')
        self.package_dir = os.path.join(home_dir, 'workspace', 'src')
        self.vision_dir = os.path.join(self.package_dir, 'Vision_')

        self.weight_path = os.path.join(
            self.vision_dir, 'segmentation', 'weights', 'rf_detr_best.pth'
        )
        self.cad_path = os.path.join(
            self.vision_dir, 'models', 'hook_model.ply'
        )

        if self.hand_side.lower() == 'left':
            camera_setting_name = 'camera_setting_left.yml'
        else:
            camera_setting_name = 'camera_setting_right.yml'

        self.camera_setting_path = os.path.join(
            self.vision_dir, 'camera', camera_setting_name
        )
        self.canonical_dir = os.path.join(
            self.vision_dir,
            'models',
            'hook_canonical_reference',
            self.hand_side.lower()
        )
        self.stable_reference_path = os.path.join(
            self.canonical_dir, 'canonical_observed_stable.ply'
        )
        self.core_reference_path = os.path.join(
            self.canonical_dir, 'canonical_observed_core.ply'
        )

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            self.vision_dir,
            'data',
            'hook_pose_estimator_node',
            self.hand_side.lower(),
            'run_' + timestamp
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.check_required_files()

        # Segmentation
        self.CONF_THRESH = 0.8
        self.min_masked_points = 500

        # Hook CAD
        self.ply_scale = 1.0
        self.junction_point = np.array(
            [908.921284, 3.411187, 33.298282], dtype=np.float64
        )

        # Global pose candidates
        # PCA 24 candidates + FPFH-RANSAC 3 candidates
        self.feature_voxel = 2.0
        self.normal_radius = 5.0
        self.feature_radius = 10.0
        self.ransac_max_correspondence = 4.5
        self.ransac_runs = 3
        self.ransac_iterations = 40000
        self.ransac_confidence = 0.999

        # Coarse CAD ICP
        self.coarse_voxel = 1.5
        self.coarse_max_correspondence = 12.0
        self.coarse_iterations = 60
        self.cad_candidate_count = 8

        # CAD refinement
        self.cad_trim_keep_ratio = 0.80
        self.cad_trim_max_correspondence = 6.0
        self.cad_trim_iterations = 50
        self.cad_trim_min_correspondences = 120
        self.cad_local_max_correspondence = 3.0
        self.cad_local_iterations = 30

        # CAD candidate check
        self.cad_max_median = 1.60
        self.cad_max_p90 = 3.20
        self.cad_max_p95 = 5.00
        self.cad_min_ir2 = 0.55
        self.cad_min_ir4 = 0.85
        self.cad_max_normal_median = 35.0

        # Canonical refinement
        self.canonical_voxel = 1.0
        self.canonical_candidate_count = 8
        self.canonical_trim_keep_ratio = 0.80
        self.canonical_trim_max_correspondence = 6.0
        self.canonical_trim_iterations = 50
        self.canonical_trim_min_correspondences = 120
        self.canonical_local_max_correspondence = 2.5
        self.canonical_local_iterations = 30

        # Final quality check
        self.final_max_stable_median = 1.50
        self.final_max_stable_p90 = 2.50
        self.final_max_stable_p95 = 4.00
        self.final_min_stable_ir2 = 0.70
        self.final_max_core_p95 = 4.00
        self.final_max_cad_p95 = 5.00

        # Ambiguity check
        self.ambiguity_median_margin = 0.15
        self.ambiguity_p90_margin = 0.25
        self.ambiguity_ir2_margin = 0.04
        self.ambiguity_translation_gap = 5.0
        self.ambiguity_rotation_gap = 5.0

        self.init_calibration()
        self.init_seg_model()
        self.init_zivid_camera()
        self.init_icp_model()
        self.init_canonical_reference()

        self.get_logger().info('Hook Pose Estimator initialized.')

    def check_required_files(self):
        required_files = [
            self.weight_path,
            self.camera_setting_path,
            self.cad_path,
            self.stable_reference_path,
            self.core_reference_path
        ]

        missing_files = []
        for path in required_files:
            if not os.path.isfile(path):
                missing_files.append(path)

        if len(missing_files) > 0:
            message = 'Required file not found:\n' + '\n'.join(missing_files)
            raise FileNotFoundError(message)

    def init_calibration(self):
        if self.hand_side.lower() == 'left':
            tx, ty, tz = -1481.770, -992.169, 1599.589
            rx, ry, rz = -120.782, -2.326, -87.213
        else:
            tx, ty, tz = -1468.350, 1004.437, 1665.196
            rx, ry, rz = -119.738, 1.822, -95.193

        rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        self.T_cam2base = np.eye(4, dtype=np.float64)
        self.T_cam2base[:3, :3] = rot.as_matrix()
        self.T_cam2base[:3, 3] = [tx, ty, tz]

    def init_seg_model(self):
        try:
            self.model = RFDETRSeg2XLarge(pretrain_weights=self.weight_path)
            if hasattr(self.model, 'optimize_for_inference'):
                self.model.optimize_for_inference()
        except Exception as e:
            self.get_logger().error(f'Error initializing Segmentation model: {e}')
            raise

    def init_zivid_camera(self):
        try:
            self.zivid_app = zivid.Application()
            self.camera = self.zivid_app.connect_camera()
            self.settings = zivid.Settings.load(self.camera_setting_path)
        except Exception as e:
            self.get_logger().error(f'Error connecting Zivid camera: {e}')
            raise

    def init_icp_model(self):
        try:
            self.gt_o3d = o3d.io.read_point_cloud(self.cad_path)

            gt_points = np.asarray(self.gt_o3d.points).copy().astype(np.float64)
            if self.ply_scale != 1.0:
                gt_points = gt_points * float(self.ply_scale)

            gt_points = gt_points - self.junction_point
            self.gt_o3d.points = o3d.utility.Vector3dVector(gt_points)

            self.gt_points = gt_points
            self.gt_tree = cKDTree(self.gt_points)
            self.gt_o3d_coarse = self.gt_o3d.voxel_down_sample(self.coarse_voxel)

            # CAD normals are used when comparing the final CAD candidates.
            gt_normal = o3d.geometry.PointCloud(self.gt_o3d)
            gt_normal.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(3.0, self.normal_radius), max_nn=60
                )
            )
            gt_normal.normalize_normals()
            self.gt_normals = np.asarray(gt_normal.normals, dtype=np.float64)

            # FPFH feature of CAD. RANSAC uses this for the global initial pose.
            self.gt_feature_pcd, self.gt_fpfh = self.prepare_fpfh_cloud(self.gt_points)

            self.get_logger().info(
                f'Loaded CAD: {len(self.gt_points)} points, '
                f'feature cloud: {len(self.gt_feature_pcd.points)} points'
            )
        except Exception as e:
            self.get_logger().error(f'Error loading ICP model: {e}')
            raise

    def init_canonical_reference(self):
        self.stable_o3d = o3d.io.read_point_cloud(self.stable_reference_path)
        self.core_o3d = o3d.io.read_point_cloud(self.core_reference_path)

        self.stable_points = np.asarray(self.stable_o3d.points, dtype=np.float64)
        self.core_points = np.asarray(self.core_o3d.points, dtype=np.float64)

        if len(self.stable_points) < 300 or len(self.core_points) < 150:
            raise RuntimeError('Canonical reference has too few points.')

        self.stable_tree = cKDTree(self.stable_points)
        self.core_tree = cKDTree(self.core_points)

        self.get_logger().info(
            f'Loaded canonical reference: stable={len(self.stable_points)}, '
            f'core={len(self.core_points)}'
        )

    def detect_hook_pose(self):
        try:
            frame = self.camera.capture_2d_3d(self.settings)
        except Exception as e:
            self.get_logger().error(f'Error capturing frame from Zivid camera: {e}')
            return False

        point_cloud = frame.point_cloud()
        xyz = point_cloud.copy_data('xyz')
        rgba = point_cloud.copy_data('rgba')
        image_pil = Image.fromarray(rgba[:, :, :3]).convert('RGB')
        image_np = np.array(image_pil)

        try:
            if torch.cuda.is_available():
                with torch.autocast('cuda', dtype=torch.float16), torch.no_grad():
                    result = self.model.predict(image_pil, threshold=self.CONF_THRESH)
            else:
                with torch.no_grad():
                    result = self.model.predict(image_pil, threshold=self.CONF_THRESH)
        except Exception as e:
            self.get_logger().error(f'Segmentation failed: {e}')
            return False

        if result.mask is None or len(result.mask) == 0:
            self.get_logger().error('No masks detected by segmentation model.')
            return False

        mask = result.mask[0]
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        mask = np.squeeze(np.asarray(mask)).astype(bool)

        valid = mask & np.isfinite(xyz).all(axis=2)
        points = xyz[valid].reshape(-1, 3).astype(np.float64)

        if len(points) < self.min_masked_points:
            self.get_logger().error('Not enough valid points in the masked point cloud.')
            return False

        self.save_segmentation_result(image_np, mask)

        # 1. Global pose search and CAD refinement
        self.get_logger().info('Running CAD registration...')
        cad_candidates = self.register_to_cad(points)
        if len(cad_candidates) == 0:
            self.get_logger().error('No CAD candidate found.')
            return False

        # 2. Canonical reference refinement
        self.get_logger().info('Running canonical reference refinement...')
        final_candidates = self.refine_to_canonical(points, cad_candidates)
        if len(final_candidates) == 0:
            self.get_logger().error('No canonical candidate found.')
            return False

        final_candidates.sort(key=self.canonical_sort_key)
        best = final_candidates[0]

        ambiguous, ambiguity_reason = self.check_ambiguity(best, final_candidates[1:])

        fail_reason = []
        if not best.quality_pass:
            fail_reason.append(
                'quality check failed '
                f'(stable median={best.stable_median_mm:.3f}, '
                f'P90={best.stable_p90_mm:.3f}, '
                f'P95={best.stable_p95_mm:.3f}, '
                f'IR@2={best.stable_ir2:.3f}, '
                f'core P95={best.core_p95_mm:.3f}, '
                f'CAD P95={best.cad_p95_mm:.3f})'
            )

        if ambiguous:
            fail_reason.append(ambiguity_reason)

        aligned_points = self.transform_points(points, best.transformation)
        self.save_alignment_results(
            image_np, xyz, mask, aligned_points, best, len(fail_reason) == 0
        )

        if len(fail_reason) > 0:
            reason = '; '.join(fail_reason)
            self.save_failed_result(reason)
            self.get_logger().error(f'Pose estimation rejected: {reason}')
            return False

        T_cam_from_hook = np.linalg.inv(best.transformation)
        T_base_from_hook = np.dot(self.T_cam2base, T_cam_from_hook)

        xyz_base = T_base_from_hook[:3, 3]
        euler_base = R.from_matrix(T_base_from_hook[:3, :3]).as_euler(
            'xyz', degrees=True
        )

        self.save_pose_result(xyz_base, euler_base)

        self.get_logger().info('============================================================')
        self.get_logger().info('Hook Pose in Robot Base Frame')
        self.get_logger().info(
            f'[X, Y, Z, Rx, Ry, Rz] = '
            f'{np.round(np.concatenate((xyz_base, euler_base)), 3)}'
        )
        self.get_logger().info('============================================================')

        return True

    # CAD registration

    def register_to_cad(self, points):
        source_points = self.voxel_downsample_points(points, self.coarse_voxel)
        source_pcd = self.make_point_cloud(source_points)

        init_list = self.build_global_init_list(source_points)
        if len(init_list) == 0:
            return []

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self.coarse_iterations
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        coarse_results = []

        for name, T_init in init_list:
            try:
                reg = o3d.pipelines.registration.registration_icp(
                    source=source_pcd,
                    target=self.gt_o3d_coarse,
                    max_correspondence_distance=self.coarse_max_correspondence,
                    init=T_init,
                    estimation_method=estimation,
                    criteria=criteria
                )

                T = np.asarray(reg.transformation, dtype=np.float64)
                aligned = self.transform_points(source_points, T)
                distances, _ = self.gt_tree.query(aligned, k=1, workers=-1)

                score = (
                    float(np.percentile(distances, 90)),
                    float(np.median(distances)),
                    float(np.percentile(distances, 95)),
                    -float(np.mean(distances <= 4.0))
                )

                coarse_results.append({
                    'name': name,
                    'T': T,
                    'fitness': float(reg.fitness),
                    'rmse': float(reg.inlier_rmse),
                    'score': score
                })
            except Exception as e:
                self.get_logger().debug(f'{name} coarse ICP failed: {e}')

        if len(coarse_results) == 0:
            return []

        # P90 -> median -> P95 -> IR@4 order
        coarse_results.sort(key=lambda x: x['score'])
        coarse_results = coarse_results[:self.cad_candidate_count]

        candidates = []

        for result in coarse_results:
            try:
                T_trim, trim_rmse = self.trimmed_icp(
                    source_points,
                    self.gt_points,
                    result['T'],
                    self.cad_trim_keep_ratio,
                    self.cad_trim_max_correspondence,
                    self.cad_trim_iterations,
                    self.cad_trim_min_correspondences
                )

                trim_candidate = self.evaluate_cad_candidate(
                    result['name'] + '+trimmed',
                    source_points,
                    T_trim,
                    result['fitness'],
                    result['rmse'],
                    trim_rmse
                )

                T_local = self.run_local_icp(
                    source_points,
                    self.gt_o3d,
                    T_trim,
                    self.cad_local_max_correspondence,
                    self.cad_local_iterations
                )

                local_candidate = self.evaluate_cad_candidate(
                    result['name'] + '+final',
                    source_points,
                    T_local,
                    result['fitness'],
                    result['rmse'],
                    trim_rmse
                )

                if self.cad_sort_key(local_candidate) < self.cad_sort_key(trim_candidate):
                    candidates.append(local_candidate)
                else:
                    candidates.append(trim_candidate)

            except Exception as e:
                self.get_logger().debug(
                    f"{result['name']} CAD refinement failed: {e}"
                )

        candidates.sort(key=self.cad_sort_key)
        return candidates

    def build_global_init_list(self, source_points):
        init_list = []

        # PCA gives 24 possible axis combinations.
        init_list.extend(self.make_pca_initializations(source_points))

        # FPFH-RANSAC is run three times with different random seeds.
        try:
            source_feature_pcd, source_fpfh = self.prepare_fpfh_cloud(source_points)

            for i in range(self.ransac_runs):
                if hasattr(o3d.utility, 'random'):
                    o3d.utility.random.seed(1000 + i)

                reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source_feature_pcd,
                    self.gt_feature_pcd,
                    source_fpfh,
                    self.gt_fpfh,
                    True,
                    self.ransac_max_correspondence,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    3,
                    [
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                            self.ransac_max_correspondence
                        )
                    ],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(
                        self.ransac_iterations,
                        self.ransac_confidence
                    )
                )

                init_list.append((
                    f'fpfh_ransac_{i:02d}',
                    np.asarray(reg.transformation, dtype=np.float64)
                ))
        except Exception as e:
            self.get_logger().warning(f'FPFH-RANSAC initialization failed: {e}')

        return self.remove_duplicate_initializations(init_list)

    def make_pca_initializations(self, source_points):
        source_center, source_axes = self.get_pca_frame(source_points)
        target_center, target_axes = self.get_pca_frame(self.gt_points)

        init_list = []
        index = 0

        for permutation in itertools.permutations(range(3)):
            P = np.eye(3)[:, permutation]

            for signs in itertools.product([-1.0, 1.0], repeat=3):
                signed_P = np.dot(P, np.diag(signs))
                rotation = np.dot(np.dot(target_axes, signed_P), source_axes.T)

                # det < 0 is reflection, not a valid 3D rotation.
                if np.linalg.det(rotation) < 0:
                    continue

                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = rotation
                T[:3, 3] = target_center - np.dot(rotation, source_center)
                init_list.append((f'pca_{index:02d}', T))
                index += 1

        return init_list

    def prepare_fpfh_cloud(self, points):
        pcd = self.make_point_cloud(points)
        pcd = pcd.voxel_down_sample(self.feature_voxel)

        if len(pcd.points) < 30:
            raise RuntimeError('Too few points for FPFH feature.')

        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=50
            )
        )
        pcd.normalize_normals()

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.feature_radius,
                max_nn=100
            )
        )

        return pcd, fpfh

    def evaluate_cad_candidate(self, name, source_points, T,
                               coarse_fitness, coarse_rmse, trimmed_rmse):
        aligned = self.transform_points(source_points, T)
        distances, nearest_indices = self.gt_tree.query(aligned, k=1, workers=-1)

        median = float(np.median(distances))
        p90 = float(np.percentile(distances, 90))
        p95 = float(np.percentile(distances, 95))
        ir1 = float(np.mean(distances <= 1.0))
        ir2 = float(np.mean(distances <= 2.0))
        ir4 = float(np.mean(distances <= 4.0))
        normal_median = self.get_normal_error(source_points, T, nearest_indices)

        quality_pass = (
            median <= self.cad_max_median and
            p90 <= self.cad_max_p90 and
            p95 <= self.cad_max_p95 and
            ir2 >= self.cad_min_ir2 and
            ir4 >= self.cad_min_ir4 and
            normal_median <= self.cad_max_normal_median
        )

        return CadCandidate(
            name=name,
            transformation=T,
            coarse_fitness=float(coarse_fitness),
            coarse_rmse_mm=float(coarse_rmse),
            trimmed_rmse_mm=float(trimmed_rmse),
            median_mm=median,
            p90_mm=p90,
            p95_mm=p95,
            ir1=ir1,
            ir2=ir2,
            ir4=ir4,
            normal_median_deg=normal_median,
            quality_pass=quality_pass
        )

    def cad_sort_key(self, candidate):
        return (
            0 if candidate.quality_pass else 1,
            candidate.p90_mm,
            candidate.median_mm,
            candidate.p95_mm,
            -candidate.ir2,
            candidate.normal_median_deg,
            candidate.trimmed_rmse_mm,
            candidate.coarse_rmse_mm
        )

    def get_normal_error(self, source_points, T, nearest_indices):
        source = self.make_point_cloud(source_points)
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=4.0, max_nn=40)
        )
        source.normalize_normals()

        source_normals = np.asarray(source.normals, dtype=np.float64)
        if source_normals.shape != source_points.shape:
            return 90.0

        rotated_normals = np.dot(source_normals, T[:3, :3].T)
        target_normals = self.gt_normals[nearest_indices]

        dot = np.sum(rotated_normals * target_normals, axis=1)
        dot = np.clip(np.abs(dot), 0.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        angle = angle[np.isfinite(angle)]

        if len(angle) == 0:
            return 90.0

        return float(np.median(angle))

    # Canonical reference refinement

    def refine_to_canonical(self, points, cad_candidates):
        source_points = self.voxel_downsample_points(points, self.canonical_voxel)
        final_candidates = []

        for cad_candidate in cad_candidates[:self.canonical_candidate_count]:
            try:
                T_trim, trim_rmse = self.trimmed_icp(
                    source_points,
                    self.stable_points,
                    cad_candidate.transformation,
                    self.canonical_trim_keep_ratio,
                    self.canonical_trim_max_correspondence,
                    self.canonical_trim_iterations,
                    self.canonical_trim_min_correspondences
                )

                trim_candidate = self.evaluate_canonical_candidate(
                    cad_candidate.name + '+canonical_trimmed',
                    source_points,
                    cad_candidate.transformation,
                    T_trim,
                    trim_rmse
                )

                T_local = self.run_local_icp(
                    source_points,
                    self.stable_o3d,
                    T_trim,
                    self.canonical_local_max_correspondence,
                    self.canonical_local_iterations
                )

                local_candidate = self.evaluate_canonical_candidate(
                    cad_candidate.name + '+canonical_final',
                    source_points,
                    cad_candidate.transformation,
                    T_local,
                    trim_rmse
                )

                if self.canonical_sort_key(local_candidate) < self.canonical_sort_key(trim_candidate):
                    final_candidates.append(local_candidate)
                else:
                    final_candidates.append(trim_candidate)

            except Exception as e:
                self.get_logger().debug(
                    f'{cad_candidate.name} canonical refinement failed: {e}'
                )

        return final_candidates

    def evaluate_canonical_candidate(self, name, source_points, T_cad, T, trimmed_rmse):
        aligned = self.transform_points(source_points, T)

        stable_distances, _ = self.stable_tree.query(aligned, k=1, workers=-1)
        core_distances, _ = self.core_tree.query(aligned, k=1, workers=-1)
        cad_distances, _ = self.gt_tree.query(aligned, k=1, workers=-1)

        stable_median = float(np.median(stable_distances))
        stable_p90 = float(np.percentile(stable_distances, 90))
        stable_p95 = float(np.percentile(stable_distances, 95))
        stable_ir1 = float(np.mean(stable_distances <= 1.0))
        stable_ir2 = float(np.mean(stable_distances <= 2.0))
        stable_ir3 = float(np.mean(stable_distances <= 3.0))

        core_median = float(np.median(core_distances))
        core_p90 = float(np.percentile(core_distances, 90))
        core_p95 = float(np.percentile(core_distances, 95))
        core_ir2 = float(np.mean(core_distances <= 2.0))

        cad_median = float(np.median(cad_distances))
        cad_p90 = float(np.percentile(cad_distances, 90))
        cad_p95 = float(np.percentile(cad_distances, 95))
        cad_ir2 = float(np.mean(cad_distances <= 2.0))

        correction = np.dot(T, np.linalg.inv(T_cad))
        correction_translation = float(np.linalg.norm(correction[:3, 3]))
        correction_rotation = self.rotation_angle_deg(correction[:3, :3])

        quality_pass = (
            stable_median <= self.final_max_stable_median and
            stable_p90 <= self.final_max_stable_p90 and
            stable_p95 <= self.final_max_stable_p95 and
            stable_ir2 >= self.final_min_stable_ir2 and
            core_p95 <= self.final_max_core_p95 and
            cad_p95 <= self.final_max_cad_p95
        )

        return CanonicalCandidate(
            name=name,
            transformation=T,
            stable_median_mm=stable_median,
            stable_p90_mm=stable_p90,
            stable_p95_mm=stable_p95,
            stable_ir1=stable_ir1,
            stable_ir2=stable_ir2,
            stable_ir3=stable_ir3,
            core_median_mm=core_median,
            core_p90_mm=core_p90,
            core_p95_mm=core_p95,
            core_ir2=core_ir2,
            cad_median_mm=cad_median,
            cad_p90_mm=cad_p90,
            cad_p95_mm=cad_p95,
            cad_ir2=cad_ir2,
            correction_translation_mm=correction_translation,
            correction_rotation_deg=correction_rotation,
            trimmed_rmse_mm=float(trimmed_rmse),
            quality_pass=quality_pass
        )

    def canonical_sort_key(self, candidate):
        return (
            0 if candidate.quality_pass else 1,
            candidate.stable_p90_mm,
            candidate.stable_median_mm,
            candidate.stable_p95_mm,
            candidate.core_p90_mm,
            -candidate.stable_ir2,
            candidate.cad_p90_mm,
            candidate.correction_rotation_deg,
            candidate.correction_translation_mm
        )

    def check_ambiguity(self, best, other_candidates):
        T_best = np.linalg.inv(best.transformation)

        for candidate in other_candidates:
            similar_quality = (
                candidate.stable_median_mm <= best.stable_median_mm + self.ambiguity_median_margin and
                candidate.stable_p90_mm <= best.stable_p90_mm + self.ambiguity_p90_margin and
                candidate.stable_ir2 >= best.stable_ir2 - self.ambiguity_ir2_margin
            )

            if not similar_quality:
                continue

            T_other = np.linalg.inv(candidate.transformation)
            translation_gap = float(np.linalg.norm(T_best[:3, 3] - T_other[:3, 3]))
            rotation_gap = self.relative_rotation_angle_deg(
                T_best[:3, :3], T_other[:3, :3]
            )

            if (translation_gap > self.ambiguity_translation_gap or
                    rotation_gap > self.ambiguity_rotation_gap):
                reason = (
                    f'ambiguous candidates: {best.name} / {candidate.name}, '
                    f'position gap={translation_gap:.3f} mm, '
                    f'rotation gap={rotation_gap:.3f} deg'
                )
                return True, reason

        return False, ''

    # ICP functions

    def trimmed_icp(self, source_points, target_points, T_init,
                    keep_ratio, max_correspondence, iterations,
                    min_correspondences):
        T = np.asarray(T_init, dtype=np.float64).copy()
        target_tree = cKDTree(target_points)
        previous_rmse = float('inf')

        for _ in range(iterations):
            transformed = self.transform_points(source_points, T)
            distances, nearest_indices = target_tree.query(
                transformed, k=1, workers=-1
            )

            valid = np.isfinite(distances) & (distances <= max_correspondence)
            valid_indices = np.flatnonzero(valid)

            if len(valid_indices) < min_correspondences:
                raise RuntimeError(
                    f'Only {len(valid_indices)} trimmed correspondences available.'
                )

            keep_count = max(
                min_correspondences,
                int(math.ceil(keep_ratio * len(valid_indices)))
            )
            keep_count = min(keep_count, len(valid_indices))

            order = np.argpartition(
                distances[valid_indices], keep_count - 1
            )[:keep_count]
            selected = valid_indices[order]

            source_selected = transformed[selected]
            target_selected = target_points[nearest_indices[selected]]

            # Find the rigid transform that best matches the selected pairs.
            delta = self.rigid_transform_svd(source_selected, target_selected)
            T = np.dot(delta, T)

            updated = self.transform_points(source_points[selected], T)
            residual = np.linalg.norm(updated - target_selected, axis=1)
            rmse = float(np.sqrt(np.mean(residual ** 2)))

            move = float(np.linalg.norm(delta[:3, 3]))
            rotate = self.rotation_angle_deg(delta[:3, :3])

            if (abs(previous_rmse - rmse) < 1e-7 and
                    move < 1e-4 and rotate < 1e-4):
                previous_rmse = rmse
                break

            previous_rmse = rmse

        return T, previous_rmse

    def run_local_icp(self, source_points, target_pcd, T_init,
                      max_correspondence, iterations):
        source = self.make_point_cloud(source_points)

        reg = o3d.pipelines.registration.registration_icp(
            source=source,
            target=target_pcd,
            max_correspondence_distance=max_correspondence,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations
            )
        )

        return np.asarray(reg.transformation, dtype=np.float64)

    # Result save / visualization

    def save_segmentation_result(self, rgb, mask):
        cv2.imwrite(
            os.path.join(self.output_dir, '01_rgb.png'),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(self.output_dir, '02_mask.png'),
            mask.astype(np.uint8) * 255
        )

        overlay = rgb.copy()
        overlay[mask] = [0, 255, 0]
        result = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
        cv2.imwrite(
            os.path.join(self.output_dir, '03_segmentation_result.png'),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        )

    def save_pose_result(self, xyz, euler):
        output_path = os.path.join(self.output_dir, 'pose_result.txt')
        with open(output_path, 'w') as f:
            f.write('status: SUCCESS\n')
            f.write('frame: base\n')
            f.write(f'x_mm: {xyz[0]:.6f}\n')
            f.write(f'y_mm: {xyz[1]:.6f}\n')
            f.write(f'z_mm: {xyz[2]:.6f}\n')
            f.write(f'euler_x_deg: {euler[0]:.6f}\n')
            f.write(f'euler_y_deg: {euler[1]:.6f}\n')
            f.write(f'euler_z_deg: {euler[2]:.6f}\n')

    def save_failed_result(self, reason):
        output_path = os.path.join(self.output_dir, 'pose_result.txt')
        with open(output_path, 'w') as f:
            f.write('status: FAILED\n')
            f.write(f'reason: {reason}\n')

    def save_alignment_results(self, rgb, xyz, mask, aligned_points, best, accepted):
        status = 'PASS' if accepted else 'REJECTED'

        self.save_alignment_image(
            self.gt_points,
            aligned_points,
            'CAD (red) vs aligned observation (green)',
            f'{status} | CAD P95={best.cad_p95_mm:.3f} mm',
            os.path.join(self.output_dir, '04_cad_alignment.png')
        )

        self.save_alignment_image(
            self.stable_points,
            aligned_points,
            'Canonical reference (red) vs observation (green)',
            f'{status} | P95={best.stable_p95_mm:.3f} mm | IR@2={best.stable_ir2:.3f}',
            os.path.join(self.output_dir, '05_canonical_alignment.png')
        )

        self.save_residual_heatmap(
            rgb,
            xyz,
            mask,
            best.transformation,
            self.stable_tree,
            f'Canonical residual | P95={best.stable_p95_mm:.3f} mm, IR@2={best.stable_ir2:.3f}',
            os.path.join(self.output_dir, '06_canonical_residual_heatmap.png')
        )

        self.save_residual_heatmap(
            rgb,
            xyz,
            mask,
            best.transformation,
            self.gt_tree,
            f'CAD residual | P95={best.cad_p95_mm:.3f} mm',
            os.path.join(self.output_dir, '07_cad_residual_heatmap.png')
        )

    def save_alignment_image(self, target_points, live_points, title, subtitle, save_path):
        target = self.voxel_downsample_points(target_points, 0.8)
        live = self.voxel_downsample_points(live_points, 0.8)

        canvas_h = 720
        canvas_w = 1800
        top_h = 110
        margin = 20
        panel_w = (canvas_w - 4 * margin) // 3
        panel_h = canvas_h - top_h - 2 * margin
        canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

        cv2.putText(canvas, title, (25, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (25, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Red: target   Green: observation   Yellow: overlap',
                    (1180, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (40, 40, 40), 1, cv2.LINE_AA)

        views = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]

        for panel_index, (view_name, axis_u, axis_v) in enumerate(views):
            x0 = margin + panel_index * (panel_w + margin)
            y0 = top_h
            panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

            target_uv = target[:, [axis_u, axis_v]]
            live_uv = live[:, [axis_u, axis_v]]
            combined = np.vstack((target_uv, live_uv))

            lower = np.percentile(combined, 0.2, axis=0)
            upper = np.percentile(combined, 99.8, axis=0)
            span = np.maximum(upper - lower, 1e-6)
            lower -= 0.06 * span
            upper += 0.06 * span
            span = upper - lower

            scale = min((panel_w - 50) / span[0], (panel_h - 70) / span[1])
            used_w = span[0] * scale
            used_h = span[1] * scale
            offset_x = (panel_w - used_w) / 2.0
            offset_y = (panel_h - used_h) / 2.0

            def project(uv):
                px = np.rint((uv[:, 0] - lower[0]) * scale + offset_x).astype(int)
                py = np.rint(
                    panel_h - 1 - ((uv[:, 1] - lower[1]) * scale + offset_y)
                ).astype(int)
                valid = (
                    (px >= 0) & (px < panel_w) &
                    (py >= 0) & (py < panel_h)
                )
                return px[valid], py[valid]

            target_x, target_y = project(target_uv)
            live_x, live_y = project(live_uv)

            target_mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
            live_mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
            target_mask[target_y, target_x] = 255
            live_mask[live_y, live_x] = 255

            kernel = np.ones((2, 2), dtype=np.uint8)
            target_mask = cv2.dilate(target_mask, kernel, iterations=1)
            live_mask = cv2.dilate(live_mask, kernel, iterations=1)

            target_only = (target_mask > 0) & (live_mask == 0)
            live_only = (live_mask > 0) & (target_mask == 0)
            overlap = (target_mask > 0) & (live_mask > 0)

            panel[target_only] = (40, 40, 220)
            panel[live_only] = (40, 190, 40)
            panel[overlap] = (0, 210, 235)

            cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1),
                          (80, 80, 80), 1)
            cv2.putText(panel, view_name, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (20, 20, 20), 2, cv2.LINE_AA)

            canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel

        cv2.imwrite(save_path, canvas)

    def save_residual_heatmap(self, rgb, xyz, mask, T, target_tree, title, save_path):
        valid = mask & np.isfinite(xyz).all(axis=2)
        points = xyz[valid].reshape(-1, 3)
        aligned = self.transform_points(points, T)
        distances, _ = target_tree.query(aligned, k=1, workers=-1)

        colors = self.residual_colors(distances)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        heat = image.copy()
        heat[valid] = colors
        blended = cv2.addWeighted(image, 0.5, heat, 0.5, 0)

        header_h = 70
        output = np.full(
            (blended.shape[0] + header_h, blended.shape[1], 3),
            255,
            dtype=np.uint8
        )
        output[header_h:] = blended

        cv2.putText(output, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(output,
                    'blue <=0.5 mm | green <=1 | yellow <=2 | orange <=4 | red >4',
                    (18, 57), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (40, 40, 40), 1, cv2.LINE_AA)

        cv2.imwrite(save_path, output)

    def residual_colors(self, distances):
        distances = np.asarray(distances, dtype=np.float64)
        colors = np.zeros((len(distances), 3), dtype=np.uint8)
        colors[distances <= 0.5] = (255, 0, 0)
        colors[(distances > 0.5) & (distances <= 1.0)] = (0, 200, 0)
        colors[(distances > 1.0) & (distances <= 2.0)] = (0, 230, 230)
        colors[(distances > 2.0) & (distances <= 4.0)] = (0, 140, 255)
        colors[distances > 4.0] = (0, 0, 255)
        return colors

    # Geometry utilities

    def make_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        return pcd

    def transform_points(self, points, T):
        points = np.asarray(points, dtype=np.float64)
        return np.dot(points, T[:3, :3].T) + T[:3, 3]

    def voxel_downsample_points(self, points, voxel_size):
        pcd = self.make_point_cloud(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points, dtype=np.float64)

    def get_pca_frame(self, points):
        points = np.asarray(points, dtype=np.float64)
        center = np.mean(points, axis=0)
        covariance = np.cov((points - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]

        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1

        return center, axes

    def rigid_transform_svd(self, source_points, target_points):
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)

        source_zero = source_points - source_center
        target_zero = target_points - target_center

        U, _, Vt = np.linalg.svd(np.dot(source_zero.T, target_zero))
        rotation = np.dot(Vt.T, U.T)

        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = np.dot(Vt.T, U.T)

        translation = target_center - np.dot(rotation, source_center)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T

    def rotation_angle_deg(self, rotation):
        value = (np.trace(rotation) - 1.0) / 2.0
        value = np.clip(value, -1.0, 1.0)
        return float(np.degrees(np.arccos(value)))

    def relative_rotation_angle_deg(self, rotation_a, rotation_b):
        return self.rotation_angle_deg(np.dot(rotation_a.T, rotation_b))

    def remove_duplicate_initializations(self, init_list):
        unique = []

        for name, T in init_list:
            duplicate = False

            for _, T_old in unique:
                trans_gap = float(np.linalg.norm(T[:3, 3] - T_old[:3, 3]))
                rot_gap = self.relative_rotation_angle_deg(
                    T[:3, :3], T_old[:3, :3]
                )

                if trans_gap < 0.5 and rot_gap < 0.5:
                    duplicate = True
                    break

            if not duplicate:
                unique.append((name, T))

        return unique


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = HookPoseEstimator()
        success = node.detect_hook_pose()
        if not success:
            node.get_logger().error('Hook pose estimation failed.')
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().error(f'Fatal error: {e}')
            node.get_logger().debug(traceback.format_exc())
        else:
            print(f'Fatal error: {e}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
