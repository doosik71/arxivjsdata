# Object-Guided Instance Segmentation With Auxiliary Feature Refinement for Biological Images

Jingru Yi, Pengxiang Wu, Hui Tang, Bo Liu, Qiaoying Huang, Hui Qu, Lianyi Han, Wei Fan, Daniel J. Hoeppner, Dimitris N. Metaxas (2021)

## 🧩 Problem to Solve

본 논문은 신경 세포(neural cell) 상호작용 연구, 식물 표현형 분석(plant phenotyping), 약물 처리 반응 측정과 같은 생물학적 이미지 분석에서 필수적인 인스턴스 분할(Instance Segmentation) 문제를 다룬다. 생물학적 이미지에서는 객체 간의 경계 대비(contrast)가 낮고 텍스처가 유사하여, 특히 서로 붙어 있거나 겹쳐 있는 객체들을 개별적으로 분리하는 것이 매우 어렵다.

기존의 Bounding Box 기반 인스턴스 분할 방법들은 먼저 객체를 박스로 캡처한 후 그 내부에서 분할을 수행하는데, 동일한 박스 영역 내에 인접한 다른 객체가 함께 포함될 경우 타겟 객체와 인접 객체를 구별하지 못하는 문제가 발생한다. 따라서 본 연구의 목표는 이러한 인접 객체의 간섭을 억제하고, 저대비 경계 영역의 세밀한 표현력을 높여 생물학적 이미지에서의 인스턴스 분할 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 검출(Detection) 단계의 정보를 분할(Segmentation) 단계의 가이드로 활용하고, 학습 과정에서 경계 영역을 집중적으로 학습시키는 보조 모듈을 도입하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **Center Keypoint 기반 객체 검출**: Anchor-box 방식의 불균형 문제와 하이퍼파라미터 설정의 어려움을 해결하기 위해, 객체의 중심점(Center point)을 검출하고 이를 통해 Bounding Box 파라미터를 직접 예측하는 방식을 채택하여 검출 정확도를 높였다.
2.  **Object-Guided Segmentation**: 검출 브랜치에서 얻은 객체 특징(Object features)을 분할 브랜치의 가이드로 재사용함으로써, 동일한 ROI(Region of Interest) 내에 존재하는 원치 않는 인접 객체를 억제하고 타겟 객체만을 정확히 분리하도록 설계하였다.
3.  **Auxiliary Feature Refinement (AFR) 모듈**: 일반적인 그리드 샘플링으로는 학습이 부족한 고주파 영역(경계선)의 특징을 보완하기 위해, 학습 과정에서 불확실성이 높은 경계 영역을 밀집하게 샘플링하여 특징을 정교화하는 보조 모듈을 제안하였다.

## 📎 Related Works

인스턴스 분할 방법론은 크게 Box-free 방식과 Box-based 방식으로 나뉜다.

*   **Box-free 방식**: DCAN, Cosine Embedding 등이 있으며, Bounding Box 없이 형태적 특성을 분석한다. 하지만 텍스처가 불균일하거나 경계가 흐릿한 경우 분할 오류가 잦으며, 특히 클러스터링 기반 방식은 객체 내부의 텍스처 변화에 민감하여 파편화된 마스크를 생성하는 한계가 있다.
*   **Box-based 방식**: Mask R-CNN, PointRend, Keypoint Graph 등이 대표적이다. 
    *   **Anchor-based (Mask R-CNN, PointRend)**: 앵커 박스의 양성/음성 샘플 불균형 문제가 있으며, 고정된 ROI 해상도로 인해 세포의 돌기(protrusion)와 같은 세밀한 구조를 놓치는 경향이 있다.
    *   **Keypoint-based (Keypoint Graph)**: 앵커 박스 문제는 해결했으나, 작은 객체의 경우 키포인트가 겹치는 문제가 발생하며, ROI 내의 인접 객체를 효과적으로 억제하지 못하는 한계가 있다.

본 논문은 이러한 기존 방식들의 한계를 극복하기 위해 중심점 기반 검출과 객체 가이드 분할, 그리고 경계 정교화 모듈을 결합하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 ResNet50을 백본으로 사용하며, 크게 **Center Keypoint-based Object Detection 브랜치**(상단)와 **Object-Guided Image Segmentation 브랜치**(하단)로 구성된다. 두 브랜치 사이에는 Skip Connection을 통해 심층적인 시맨틱 정보와 얕은 층의 세부 정보를 결합한다.

### 1. Center Keypoint-Based Object Detection
검출 헤드(DecHead)는 다음 세 가지 맵을 출력한다.
*   **Center Heatmap**: 객체의 중심점에 2D Gaussian blob을 배치하여 생성한다. 배경 픽셀에 대한 페널티를 조절하기 위해 변형된 Focal Loss를 사용하여 학습한다.
    $$\mathcal{L}_{heat} = -\frac{1}{N} \begin{cases} (1-p_i)^\alpha \log(p_i) & \text{if } y_i=1 \\ (1-y_i)^\beta (p_i)^\alpha \log(1-p_i) & \text{otherwise} \end{cases}$$
*   **Width-Height Map**: 중심점에서의 객체 너비($w$)와 높이($h$)를 예측하며, Smooth $L_1$ 손실 함수를 사용한다.
*   **Offset Map**: 입력 이미지와 출력 맵 사이의 다운샘플링으로 인한 이산화 오차를 보정하기 위해 오프셋 $o$를 예측하며, 역시 Smooth $L_1$ 손실을 사용한다.

최종 Bounding Box는 중심점 위치 $\bar{c} = (c_u + o_u, c_v + o_v)$와 너비, 높이를 결합하여 디코딩된다.

### 2. Object-Guided Segmentation
검출된 Bounding Box를 기반으로 특징 맵에서 ROI 패치를 크롭하여 분할을 수행한다.
*   **Object-Guidance**: 얕은 층의 특징($c_1, c_2$)은 세부 형태 정보는 많지만 인접 객체와 구분이 어렵다. 이를 해결하기 위해 검출 브랜치의 깊은 층 특징($c_3, c_5$)을 가이드로 활용하여 타겟 객체 외의 영역을 억제한다.
*   **Instance Normalization**: ROI 패치들의 크기와 비율이 다양하므로, Batch Normalization 대신 Instance Normalization을 사용하여 정규화한다.
*   **손실 함수**: 이진 교차 엔트로피(Binary Cross-Entropy, BCE) 손실 $\mathcal{L}_{seg}$를 사용하여 마스크를 최적화한다.

### 3. Auxiliary Feature Refinement (AFR) Module
학습 과정에서만 동작하는 보조 모듈로, 경계 영역의 정밀도를 높인다.
*   **불확실성 맵(Uncertainty Map)**: 예측된 마스크 $x$를 이용하여 $\mathcal{x}' = -|2x-1|$ 식을 통해 확률이 0.5 근처인 불확실한 영역(경계선)을 식별한다.
*   **비균일 샘플링(Non-Uniform Sampling)**: 불확실성 맵에서 값이 높은(가장 불확실한) 상위 $\beta N$개의 지점을 부동 소수점 좌표로 샘플링한다.
*   **특징 정교화**: 샘플링된 지점의 특징을 추출하여 $1 \times 1$ 컨볼루션 층으로 구성된 RefineHead를 통해 정교화하며, 정답 마스크의 보간된 라벨을 사용하여 BCE 손실 $\mathcal{L}_{ref}$로 학습한다.

## 📊 Results

### 실험 설정
*   **데이터셋**: Plant Phenotyping(식물 잎), Neural Cell(신경 세포), DSB2018(세포 핵)의 3가지 생물학적 데이터셋을 사용하였다.
*   **평가 지표**: 검출 성능은 $\text{AP}_{dec}$, 분할 성능은 $\text{AP}_{seg}$ (IoU 임계값 0.5~0.95 평균)를 사용하였다.
*   **비교 대상**: DCAN, Cosine Embedding (Box-free), Mask R-CNN, PointRend, Keypoint Graph (Box-based).

### 주요 결과
1.  **정량적 성능**: 제안 방법(Ours W. Refine)은 모든 데이터셋에서 기존 방법론보다 우수한 성능을 보였다. 특히 돌기 구조가 많은 Neural Cell과 Plant 데이터셋에서 $\text{AP}_{seg}$ 향상 폭이 컸다.
2.  **정성적 분석**: 
    *   **PointRend** 대비: PointRend는 경계선을 렌더링하여 세밀하지만 마스크가 파편화되거나 구멍이 생기는 문제가 있었으나, 제안 방법은 더 완전하고 매끄러운 마스크를 생성하였다.
    *   **Keypoint Graph** 대비: 동일 ROI 내의 인접 객체를 효과적으로 억제하여 오검출을 줄였으며, 특히 작은 객체 검출 능력이 향상되었다.
3.  **AFR 모듈의 효과**: $\text{AP}_{seg}$ 분석 결과, AFR 모듈을 적용했을 때(W. Refine) 특히 고주파 영역인 경계선 표현력이 눈에 띄게 개선됨을 확인하였다.
4.  **추가 검증**: Cell Tracking Challenge(CTC)와 Leaf Segmentation Challenge(LSC)에서도 준수한 성능을 기록하며 범용성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 검출과 분할을 독립적인 단계로 보지 않고, 검출 단계의 고수준 시맨틱 정보를 분할 단계의 '가이드'로 활용함으로써 ROI 내 객체 간 간섭 문제를 효율적으로 해결하였다. 또한, 모든 픽셀을 동일하게 학습시키는 대신, 학습 시에만 불확실한 경계 영역을 집중적으로 샘플링하는 AFR 모듈을 통해 추론 속도 저하 없이 분할 정밀도를 높인 점이 인상적이다.

### 한계 및 미해결 과제
1.  **파편화된 분할 (Fragmentary Segmentations)**: 대비가 극도로 낮거나 형태가 비정상적인 신경 세포의 경우, 여전히 일부 마스크가 파편화되어 생성되는 현상이 발견되었다. 저자들은 이를 해결하기 위해 향후 Shape Prior(형태 사전 정보)를 도입할 계획임을 밝혔다.
2.  **데이터 의존성 (Annotation Hungry)**: 학습 데이터의 수가 매우 적은 경우(예: LSC의 일부 데이터셋), 대규모 파라미터를 가진 딥러닝 모델 특성상 성능이 급격히 저하되는 모습을 보였다. 이는 Few-shot learning이나 Unsupervised learning의 도입이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 생물학적 이미지의 저대비 경계와 객체 겹침 문제를 해결하기 위해 **중심점 기반 검출 $\rightarrow$ 객체 가이드 분할 $\rightarrow$ 보조 경계 정교화(AFR)**로 이어지는 파이프라인을 제안한다. 특히 검출 특징을 활용해 인접 객체를 억제하고, 학습 단계에서만 경계 영역을 밀집 샘플링하여 정밀도를 높였다. 이 연구는 세밀한 구조(세포 돌기, 식물 잎줄기 등)를 보존해야 하는 의료 및 생물학적 이미지 분석 분야의 인스턴스 분할 성능을 한 단계 높이는 데 기여할 가능성이 크다.