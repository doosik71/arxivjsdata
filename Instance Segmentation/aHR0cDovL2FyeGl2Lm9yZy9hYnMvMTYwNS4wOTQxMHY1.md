# End-to-End Instance Segmentation with Recurrent Attention

Mengye Ren, Richard S. Zemel

## 🧩 Problem to Solve

인스턴스 분할(Instance Segmentation)은 이미지 내의 개별 객체 인스턴스를 픽셀 수준에서 식별하고 분할하는 핵심 컴퓨터 비전 문제입니다. 이는 자율 주행, 이미지 캡셔닝 등 다양한 응용 분야에서 중요하지만, 다음과 같은 도전 과제를 안고 있습니다:

- **높은 출력 차원**: 표준 FCN(Fully Convolutional Networks)은 모든 인스턴스 레이블을 한 번에 출력하는 데 어려움이 있습니다.
- **복잡한 기존 방법**: 기존의 그래프 모델 기반 접근 방식은 파이프라인이 복잡하고, 시간이 많이 소요되며, 엔드투엔드 학습이 어렵습니다.
- **가려짐(Occlusion) 처리**: 객체 간의 가려짐은 인스턴스 분할의 주요 난제이며, 바텀업(bottom-up) 방식이나 NMS(Non-Maximal Suppression)로는 처리하기 어렵습니다.
- **개수 세기(Counting)**: 인스턴스 분할과 밀접하게 관련된 객체 개수 세기 역시 단독으로도 어려운 문제입니다.

이 논문은 이러한 문제들을 해결하기 위해 인간의 인지 과정에서 영감을 받은 엔드투엔드 모델을 제안합니다.

## ✨ Key Contributions

- 시각적 어텐션 메커니즘을 사용하는 엔드투엔드 순환 신경망(RNN) 아키텍처를 제안하여 인스턴스 분할을 수행합니다.
- 외부 메모리를 통해 이전에 분할된 객체 정보를 추적하고, 이를 바탕으로 가려진 객체를 추론하여 동적 NMS(Non-Maximal Suppression)를 가능하게 합니다.
- 박스 네트워크와 분할 네트워크를 함께 학습시켜 관심 영역을 순차적으로 생성하고, 각 영역 내의 지배적인 객체를 분할합니다.
- 객체 개수 세기를 인스턴스 분할과 함께 공동으로 학습하며, 순환 과정의 자동 종료 조건을 설정합니다.
- 인스턴스 매칭을 위한 `matching IoU loss`, 미분 가능한 `soft box IoU loss`, 그리고 자동 종료를 위한 `monotonic score loss`를 제안합니다.
- CVPPP, KITTI, Cityscapes 데이터셋에서 경쟁력 있는 결과를 달성하며, 특히 이전 RNN 기반 방법론 대비 상당한 성능 향상을 보였습니다.

## 📎 Related Works

- **Detector-based approaches**: 객체 탐지기(예: DPM, RCNN)를 기반으로 바운딩 박스 내에서 분할을 수행하는 방법들 [5, 43, 17, 18, 25, 24, 10].
- **Graphical model approaches**: 생성적 그래프 모델(예: RBM, CRF)을 사용하여 인스턴스 및 픽셀 간의 의존성 구조를 모델링하는 방법들 [11, 39, 45]. 복잡성과 긴 실행 시간이 단점입니다.
- **Fully convolutional approaches**: FCN [27]을 사용하여 픽셀 레이블을 직접 예측합니다. 객체 제안(object proposal)을 생성하거나 바텀업 병합(bottom-up merging)에 의존하는 방식 [33, 34, 9, 26, 41, 2, 19, 22]. 일부는 인스턴스 인식을 위해 각도 맵(angle map) 등을 활용합니다.
- **RNN approaches**: 엔드투엔드 RNN을 활용하여 객체 탐지 및 분할을 수행하는 최근 연구들 [40, 32, 36]. 순차적 분해 아이디어를 사용하며, 본 연구는 기존 RNN 방식의 한계를 극복하고 피드백 루프를 도입합니다.

## 🛠️ Methodology

이 모델은 순환 신경망(RNN)과 어텐션 메커니즘을 활용하여 객체 인스턴스를 순차적으로 하나씩 분할합니다.

- **입력 전처리(Input Pre-processing)**:
  - 사전 학습된 FCN을 사용하여 입력 이미지에서 1채널 전경 분할(foreground segmentation) 마스크와 8채널 각도 맵(angle map, 객체 중심에 대한 상대 각도)을 생성하여 객체 경계에 대한 상세 정보를 제공합니다.
- **주요 구성 요소**:
    1. **A) 외부 메모리(External memory)**: 이전에 분할된 객체의 누적 마스크 $c_t = \max(c_{t-1}, y_{t-1})$를 저장하고, 이를 전처리된 입력 $x$와 결합($d_t = [c_t, x]$)하여 다음 관심 영역을 결정하는 데 활용합니다. 가려진 객체에 대한 추론에 도움을 줍니다.
    2. **B) 박스 제안 네트워크(Box proposal network)**:
        - 외부 메모리에서 얻은 $d_t$를 CNN에 통과시켜 특징 맵 $u_t$를 생성합니다.
        - Glimpse LSTM과 "소프트 어텐션(soft-attention)" 메커니즘을 사용하여 공간 정보를 추출합니다.
        - 예측된 바운딩 박스 좌표 ($g_X, g_Y, \delta_X, \delta_Y$), 스케일링 팩터 $\gamma$, 그리고 가우시안 보간 커널($\mu_{X,Y}, \sigma_{X,Y}$) 파라미터를 사용하여 이미지 패치 $p_t$를 추출합니다.
    3. **C) 분할 네트워크(Segmentation network)**:
        - 추출된 패치 $p_t$를 입력으로 받아 CNN을 통해 특징 맵 $v_t$를 생성합니다.
        - DeconvNet과 스킵 연결을 사용하여 패치 수준의 분할 예측 $\tilde{y}_t$를 만듭니다.
        - $\tilde{y}_t$를 가우시안 필터의 전치(transpose)를 사용하여 원래 이미지 크기의 분할 $y_t$로 재투영합니다. 이때 $\gamma$로 신호를 증폭하고 $\beta$로 외부 픽셀을 억제합니다.
    4. **D) 스코어링 네트워크(Scoring network)**:
        - 박스 네트워크와 분할 네트워크의 은닉 상태($z_{t, \text{end}}, v_t$)를 입력으로 받아 현재 인스턴스에 대한 신뢰도 점수 $s_t \in [0,1]$를 예측합니다.
        - $s_t < 0.5$일 때 순차적 처리 과정을 종료합니다.
- **손실 함수(Loss Functions)**:
  - **총 손실**: $L = L_y(y, y^*) + \lambda_b L_b(b, b^*) + \lambda_s L_s(s, s^*)$ (여기서 $\lambda_b = \lambda_s = 1$).
  - **(a) Matching IoU 손실 ($L_y$)**: 예측된 인스턴스와 실제(ground-truth) 인스턴스 간의 최대 가중 이분 매칭(maximum-weighted bipartite graph matching, 헝가리안 알고리즘)을 기반으로 `softIOU`를 사용하여 계산됩니다. 이는 순서에 둔감하며, 오탐(false positive)과 미탐(false negative)을 모두 페널티합니다.
        $$ L_y(y, y^*) = -\text{mIOU}(y, y^*) = -\frac{1}{N} \sum_{i,j} M_{i,j} \mathbf{1}[\text{match}(y_i) = y^*_j] $$
        $$ M_{i,j} = \text{softIOU}(y_i, y^*_j) \equiv \frac{\sum y_i \cdot y^*_j}{\sum y_i + y^*_j - y_i \cdot y^*_j} $$
  - **(b) Soft Box IoU 손실 ($L_b$)**: 예측된 박스와 실제 박스 간의 IoU를 미분 가능하게 만든 버전입니다. 가우시안 필터를 통해 상수 패치를 원본 이미지에 재투영하고, 패딩된 실제 바운딩 박스와의 mIOU를 계산합니다.
  - **(c) Monotonic Score 손실 ($L_s$)**: 스코어 출력이 단조적으로 감소하도록 장려하여 자동 종료를 용이하게 합니다. 실제 점수 $s_t^*$가 1인 경우 이전 스코어의 하한선과 비교하고, 0인 경우 이후 스코어의 상한선과 비교합니다.
- **학습 절차(Training Procedure)**:
  - **부트스트랩 학습(Bootstrap training)**: 박스 및 분할 네트워크는 처음에는 실제 값(ground-truth)으로 사전 학습된 후, 나중에는 모델 예측값으로 대체됩니다.
  - **스케줄드 샘플링(Scheduled sampling)**: 학습 과정에서 외부 메모리의 입력으로 실제 분할 또는 이전 단계의 모델 출력을 확률적으로 선택하여 사용합니다. 학습이 진행됨에 따라 실제 값에 대한 의존도를 점진적으로 줄여 테스트 시나리오와 유사하게 만듭니다.

## 📊 Results

- **CVPPP 잎 분할 (Leaf Segmentation)**:
  - SBD(Symmetric Best Dice) 84.9, |DiC|(Absolute Difference in Count) 0.8로 이전 최고 성능(SBD 74.4, |DiC| 1.1)을 크게 상회했습니다.
  - 데이터셋 크기가 작아 오버피팅을 방지하기 위해 FCN 전처리 없는 단순화된 모델을 사용했습니다.
- **KITTI 차량 분할 (Car Segmentation)**:
  - MWCov(Mean Weighted Coverage) 80.0, MUCov(Mean Unweighted Coverage) 66.9를 기록하여 [46, 45]보다 우수했습니다.
  - [41]의 경우 깊이 정보(depth information) 및 "인스턴스 융합(instance fusion)"을 사용하여 본 모델보다 약간 높은 MWCov 79.7, MUCov 75.8을 달성했습니다.
- **Cityscapes 인스턴스 분할 (Multi-class Instance Segmentation)**:
  - 전체 클래스에서 AP 9.5, 차량 클래스에서 AP 27.5를 달성하여 경쟁력 있는 성능을 보였습니다. 특히 50m 및 100m 거리에서 AP 성능이 우수했습니다.
- **MS-COCO 얼룩말 이미지 (MS-COCO Zebra Images)**:
  - MWCov 69.2, MUCov 64.2, |DiC| 0.79를 기록하여 개수 세기에서 기존 탐지 기반 및 연관 서비타이징(associative-subitizing) 방법보다 우수했습니다.
- **KITTI 검증셋에 대한 어블레이션 연구(Ablation Studies)**:
  - **전처리 없음(No Pre-processing)**: 성능이 MWCov 55.6으로 크게 하락하여 전경 분할 및 각도 맵의 중요성을 확인했습니다.
  - **박스 네트워크 없음(No Box Net)**: MWCov 57.0으로 성능이 크게 하락하여 지역화의 중요성을 보여주었습니다.
  - **각도 맵 없음(No Angles)**: MWCov 71.2로 성능이 감소하여 각도 맵의 기여를 입증했습니다.
  - **스케줄드 샘플링 없음(No Scheduled Sampling)**: MWCov 73.6으로 소폭 하락하여 학습 절차의 효과를 확인했습니다.
  - **Glimpse 반복 횟수(Fewer Iterations)**: 반복 횟수가 증가함에 따라 성능이 향상됨(Iter-1: 64.1 -> Iter-5: 75.1)을 보였습니다.

## 🧠 Insights & Discussion

- **강점**:
  - 제안된 탑다운(top-down) 어텐션 기반 추론 방식은 자전거와 같이 분리된 구성 요소를 하나의 객체로 인식하고, 심하게 가려진 장면에서도 인스턴스를 효과적으로 분할합니다.
  - 최종 분할 마스크를 직접 출력하므로 다른 방법에서 흔히 필요한 후처리(post-processing) 과정이 필요 없습니다.
  - 시각적 어텐션을 통해 인스턴스의 중요도를 학습하며, 초기에는 공간적 순서(예: 왼쪽에서 오른쪽)로 객체를 처리하다가 점차 신뢰도 기반의 정교한 순서로 전환됩니다.
  - 적은 수의 파라미터로도 강력한 성능을 보이며, 비교적 작은 데이터셋으로도 효과적인 학습이 가능합니다.
- **한계**:
  - 다운샘플링으로 인해 멀리 있는 객체를 누락하거나 과소 분할(under-segmentation)하는 경향이 있습니다.
  - "세 번째 사지"와 같은 고차원적인 추론(higher-order reasoning)이 부족할 수 있습니다.
- **향후 연구**:
  - 본 모델을 바텀업 병합 방법 [26, 41]이나 고차원 그래프 모델 [29, 11]과 결합하여 이러한 한계를 보완할 수 있습니다.
  - MS-COCO와 같은 다중 클래스 인스턴스 분할 및 더욱 구조화된 장면 이해로 확장을 모색합니다.

## 📌 TL;DR

인스턴스 분할은 개별 객체를 픽셀 단위로 정확하게 구분하는 어려운 문제이며, 특히 가려짐과 기존 방법의 복잡성이 주요 장애물이었습니다. 이 논문은 인간의 개수 세기 과정에서 영감을 받은 **엔드투엔드 순환 신경망(RNN) 기반의 어텐션 모델**을 제안합니다. 이 모델은 **외부 메모리**를 활용하여 가려진 객체를 추론하고, **박스 및 분할 네트워크**를 통해 관심 영역 내의 객체를 순차적으로 분할합니다. 또한, **매칭 IoU 손실, 소프트 박스 IoU 손실, 단조 스코어 손실** 등의 새로운 손실 함수를 통해 엔드투엔드 학습을 가능하게 하고 자동 종료 기능을 통합합니다. 결과적으로 CVPPP, KITTI, Cityscapes 데이터셋에서 SOTA에 가까운 성능을 달성하며, 효율적인 가려짐 처리와 후처리 없는 직접 분할 결과를 제공하는 것이 핵심 기여입니다.
