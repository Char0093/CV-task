"""Match distinctive ORB features between the two mug images."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / "images" / "reference.png"
QUERY_PATH = ROOT / "images" / "query.png"
OUTPUT_PATH = ROOT / "results" / "matches.png"


def main() -> None:
    reference = cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_COLOR)
    query = cv2.imread(str(QUERY_PATH), cv2.IMREAD_COLOR)
    if reference is None or query is None:
        raise FileNotFoundError(
            "Could not read the mug image pair. Expected files at "
            f"{REFERENCE_PATH} and {QUERY_PATH}."
        )

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference_gray, None
    )
    query_keypoints, query_descriptors = orb.detectAndCompute(query_gray, None)

    if reference_descriptors is None or query_descriptors is None:
        raise RuntimeError("ORB could not find enough features in both images.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    neighbors = matcher.knnMatch(reference_descriptors, query_descriptors, k=2)
    ratio_matches = [
        best
        for pair in neighbors
        if len(pair) == 2
        for best, second in [pair]
        if best.distance < 0.75 * second.distance
    ]

    if len(ratio_matches) < 4:
        raise RuntimeError(
            "Fewer than four reliable matches were found; "
            "a homography cannot be estimated."
        )

    source_points = np.float32(
        [reference_keypoints[match.queryIdx].pt for match in ratio_matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [query_keypoints[match.trainIdx].pt for match in ratio_matches]
    ).reshape(-1, 1, 2)

    _, inlier_mask = cv2.findHomography(
        source_points, destination_points, cv2.RANSAC, 5.0
    )
    if inlier_mask is None:
        raise RuntimeError("RANSAC could not estimate a consistent transform.")

    inliers = [
        match for match, is_inlier in zip(ratio_matches, inlier_mask.ravel())
        if is_inlier
    ]
    visualization = cv2.drawMatches(
        reference,
        reference_keypoints,
        query,
        query_keypoints,
        inliers[:100],
        None,
        matchColor=(70, 210, 110),
        singlePointColor=(150, 150, 150),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    header = np.full((76, visualization.shape[1], 3), 248, dtype=np.uint8)
    cv2.putText(
        header,
        f"ORB feature matches  |  ratio-test: {len(ratio_matches)}  |  "
        f"RANSAC inliers: {len(inliers)}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (45, 55, 70),
        2,
        cv2.LINE_AA,
    )
    result = np.vstack((header, visualization))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), result):
        raise OSError(f"Could not write result image to {OUTPUT_PATH}.")

    print(f"Reference keypoints: {len(reference_keypoints)}")
    print(f"Query keypoints: {len(query_keypoints)}")
    print(f"Ratio-test matches: {len(ratio_matches)}")
    print(f"RANSAC inliers: {len(inliers)}")
    print(f"Saved visualization: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
