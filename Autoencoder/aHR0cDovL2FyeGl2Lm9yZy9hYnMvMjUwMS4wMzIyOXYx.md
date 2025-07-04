# GAUSSIANMASKEDAUTOENCODERS
Jathushan Rajasegaran, Xinlei Chen, Rulilong Li, Christoph Feichtenhofer, Jitendra Malik, Shiry Ginosar

## Problem to Solve

기존의 Masked Autoencoders (MAE)와 같은 재구성 기반 자기 지도 학습(Self-supervised Learning) 프레임워크는 우수한 의미론적 추상화(semantic abstractions)를 학습하지만, 명시적인 공간적 이해(spatial awareness)를 위해 훈련되지 않습니다. 시각적 추론에는 데이터의 고수준 의미론적 추상화뿐만 아니라 3D 공간에서의 객체 및 관계에 대한 공간적 이해가 필수적입니다. 이 논문은 MAE의 강점을 유지하면서도 이미지의 공간적 구조를 동시에 학습하는 방법을 탐구합니다.

## Key Contributions

*   **Gaussian Masked Autoencoder (GMAE)** 제안: MAE를 확장하여 중간 3D 가우시안 기반 표현을 도입함으로써 의미론적 추상화와 공간적 이해를 동시에 학습합니다.
*   **최초의 가우시안 프리미티브 활용**: 단일 장면 재구성을 위한 최적화 기반 프레임워크를 넘어, 이미지 표현 학습 프레임워크에 가우시안 프리미티브를 활용한 최초의 연구입니다.
*   **제로샷 공간 이해 능력**: 이미지 재구성 훈련만으로도 figure-ground 분리(객체-배경 분리), 이미지 레이어링, 에지 감지 등 다양한 제로샷(zero-shot) 공간 이해 능력을 가능하게 합니다.
*   **MAE와 동등한 성능**: 기존 MAE와 유사하게 이미지 분류 및 객체 감지 등 고수준 의미론적 작업에서 경쟁력 있는 표현 학습 성능을 유지합니다.
*   **효율적인 학습**: 표준 MAE 학습 대비 미미한 오버헤드(연산 시간 1.5% 증가)만으로 추가적인 공간 이해 능력을 제공합니다.
*   **고품질 재구성**: 가우시안의 비균일성 및 동적 배치 능력 덕분에 MAE보다 더 높은 품질의 이미지 재구성(낮은 rFID 점수)을 달성합니다.

## Methodology

`GMAE`는 `ViT` 기반 인코더, 경량 디코더, 그리고 미분 가능한 렌더러로 구성됩니다.

1.  **입력 처리**: 입력 이미지를 `N`개의 패치로 분할하고, `r` 비율로 무작위 마스킹하여 `n`개의 가시 패치를 생성합니다.
2.  **인코더 단계**: `ViT` 인코더는 가시 패치만을 입력받아 잠재 임베딩 $x_i \in \mathbb{R}^{d_{enc}}$로 인코딩합니다.
3.  **디코더 단계**:
    *   디코더는 `k`개의 학습 가능한 쿼리 토큰 $q_j \in \mathbb{R}^{d_{dec}}$와 인코딩된 잠재 임베딩의 투영값 $\hat{x}_i \in \mathbb{R}^{d_{dec}}$을 결합한 $X_{dec} = \{\hat{x}_i\}_{i=1}^n \cup \{q_j\}_{j=1}^k$를 입력받습니다.
    *   디코더는 각 쿼리 토큰에 대해 14차원 벡터 $g_j=\{p,s,\phi,r,o\}$로 매개변수화된 `k`개의 3D 가우시안을 예측합니다. 여기서 $p$는 3D 중심 위치, $s$는 스케일, $\phi$는 회전 쿼터니언, $r$은 색상, $o$는 불투명도를 나타냅니다.
4.  **렌더링 단계**: 예측된 가우시안들은 고정된 카메라 투영을 사용하여 스플래팅(splatting) 미분 가능 렌더러를 통해 이미지로 렌더링됩니다. 가우시안의 크기는 $c \cdot \text{sigmoid}(s)$를 사용하여 제한됩니다.
5.  **손실 계산**: 재구성된 이미지와 원본 이미지 간의 평균 제곱 오차(MSE) 손실이 마스킹된 픽셀 영역에만 적용되어 전체 모델을 엔드투엔드(end-to-end)로 학습합니다.
    *   특이점: 일반적인 가우시안 스플래팅과 달리, `GMAE`는 포인트 클라우드 초기화 없이 이미지 재구성을 통해 모든 가우시안 속성을 직접 학습합니다.

## Results

*   **지도 학습 작업**:
    *   **ImageNet 분류**: 400 에폭 훈련 시 `GMAE`는 83.2%의 Top-1 정확도를 달성하여 `MAE`와 유사한 성능을 보였습니다. 가우시안 수가 증가함에 따라 성능이 향상되었으며, 256개 이상에서는 포화 상태를 보였습니다.
    *   **COCO 객체 감지 및 분할**: `GMAE`는 `MAE`와 비슷한 $AP_{box}$ 50.2, $AP_{mask}$ 44.5를 기록하며 지도 학습 기반 사전 훈련을 크게 능가했습니다.
*   **비지도 학습 작업 (공간 이해)**:
    *   **재구성 품질**: `GMAE`는 ImageNet 검증 세트에서 `MAE`(98.12 rFID)보다 훨씬 낮은 89.45 rFID(낮을수록 좋음)를 달성하여 우수한 재구성 품질을 입증했습니다. 이는 가우시안의 비균일한 분포가 고주파수 정보를 잘 모델링하기 때문입니다.
    *   **제로샷 Figure-Ground 분리**: 가우시안을 깊이 값으로 정렬한 후 레이어별로 분리하여 전경-배경 분리를 수행했습니다. PASCAL 데이터셋에서 `MAE`를 포함한 다른 제로샷/소수샷(few-shot) 베이스라인보다 우수한 성능을 보였습니다.
    *   **제로샷 에지 감지**: 가우시안 레이어 간의 깊이 불연속성을 통해 에지를 감지했습니다. BSDS500 데이터셋에서 합리적인 성능을 보여주었으며, 레이어 수를 조절하여 다양한 입도의 에지를 감지할 수 있음을 입증했습니다.
*   **정성적 결과**:
    *   **가우시안 분포**: 가우시안은 이미지 정보 밀도 및 고주파수 세부 사항에 따라 동적으로 위치와 크기를 조절하여 고주파수 영역을 고화질로 모델링할 수 있습니다.
    *   **크기 대 깊이 상관관계**: 평균적으로 스케일 값이 큰 가우시안(저주파수 정보)은 카메라에 더 가깝게 위치하고, 스케일 값이 작은 가우시안(고주파수 세부 사항)은 카메라에서 더 멀리 위치하는 명확한 경향을 보였습니다. 이러한 특성은 레이어링 효과 및 제로샷 성능의 근거가 됩니다.