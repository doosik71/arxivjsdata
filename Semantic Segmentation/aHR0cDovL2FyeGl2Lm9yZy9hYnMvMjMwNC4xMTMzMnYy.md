# Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model
Yizhe Zhang, Tao Zhou, Shuo Wang, Peixian Liang, Danny Z. Chen

## 🧩 Problem to Solve
최근 개발된 Segment Anything Model (SAM)은 컴퓨터 비전 작업을 위한 범용 분할 모델로, 방대한 데이터셋으로 학습되어 자연 이미지에서 다양한 객체의 분할 결과를 생성할 수 있습니다. 그러나 SAM은 의료 영상 분할과 같이 도메인 전문 지식이 필요한 작업에서는 충분히 높은 품질의 분할 결과를 제공하지 못하는 한계가 있습니다. 이 연구는 SAM이 의료 영상 데이터를 직접적으로 고품질로 분할하지는 못하더라도, SAM이 생성하는 마스크, 특징, 안정성 점수 등을 활용하여 더 나은 의료 영상 분할 모델을 구축하고 훈련하는 방법을 탐구합니다. 특히, 보편적으로 사용되는 의료 영상 분할 모델(예: U-Net)의 입력 데이터를 SAM을 활용하여 증강하는 방법을 제안합니다.

## ✨ Key Contributions
*   범용 분할 기반 모델인 SAM이 다운스트림 분할 작업을 위한 어텐션(사전) 맵을 제공할 수 있음을 확인했습니다.
*   간단하고 새로운 방법인 SAMAug를 제안하여 SAM의 분할 출력과 원본 이미지 입력을 결합함으로써, 다운스트림 의료 영상 분할 모델을 위한 SAM 증강 입력 이미지를 생성합니다.
*   세 가지 의료 영상 분할 작업에서 CNN 및 Transformer 기반 분할 모델 모두에 대해 제안된 SAMAug 방법의 효과를 포괄적인 실험을 통해 입증했습니다.

## 📎 Related Works
*   **데이터 증강 (Data Augmentation, DA)**: 의료 영상 분할 모델 훈련에 널리 사용되는 기법으로, 기존 샘플의 새로운 뷰를 합성하는 것을 목표로 합니다. SAMAug는 기존의 수동 설계 변환(예: 회전, 크롭)과 달리, 분할 기반 모델을 사용하여 원본 이미지를 증강하여 의미적으로 유용한 구조를 입력에 추가합니다.
*   **이미지 개선 (Image Enhancement, IE)**: SAMAug는 분할 기반 모델의 의미론적 구조를 추가하여 이미지를 개선합니다. 기존 IE 방법들이 디블러링, 노이즈 감소와 같은 저수준 작업에 중점을 두는 반면, SAMAug는 고수준 구조를 추가하여 후속 의료 영상 분할 모델에 더 나은 의미론적 정보를 제공합니다.
*   **최근 SAM 관련 방법**: SAM 출시 이후, SAM을 의료 영상 분석에 활용하려는 다양한 시도가 있었습니다. 대부분의 연구에서 SAM 단독으로는 의료 영상 분할 작업에 만족스러운 결과를 제공하지 못함이 밝혀졌습니다. Ma et al. [13]은 레이블된 이미지를 사용하여 SAM을 파인튜닝하는 방법을 제안했고, Wu et al. [25]는 추가 레이어를 사용하여 SAM을 의료 영상 분할 작업에 적응시키는 방법을 제안했습니다. SAMAug는 이러한 파인튜닝 및 적응 방법보다 모델 훈련 시 계산 및 메모리 비용 측면에서 더 효율적입니다.

## 🛠️ Methodology
제안하는 SAMAug 방법은 SAM의 출력을 활용하여 의료 영상 모델의 입력 데이터를 증강하는 방식으로 작동합니다.

1.  **분할 및 경계 사전 맵(Prior Maps) 생성**:
    *   SAM의 그리드 프롬프트(grid prompt) 설정을 사용하여 주어진 의료 영상에 대해 다수의 분할 마스크를 생성합니다.
    *   각 마스크의 **안정성 점수(stability score)**를 활용하여 **분할 사전 맵(segmentation prior map)**을 생성합니다. 안정성 점수가 높을수록 해당 마스크가 더 신뢰할 수 있다고 간주됩니다.
    *   각 분할 마스크의 외부 경계(exterior boundary)를 추출하여 **경계 사전 맵(boundary prior map)**을 생성합니다.

2.  **입력 이미지 증강**:
    *   생성된 분할 사전 맵과 경계 사전 맵을 원본 이미지 $x$에 추가하여 입력 이미지를 증강합니다.
    *   원본 이미지가 흑백인 경우, 첫 번째 채널은 흑백 원본 이미지, 두 번째 채널은 분할 사전 맵, 세 번째 채널은 경계 사전 맵으로 구성된 3채널 이미지를 생성합니다.
    *   증강된 이미지는 $x_{\text{aug}} = \text{Aug}(\text{prior}_{\text{seg}}, \text{prior}_{\text{boundary}}, x)$와 같이 표현됩니다.

3.  **SAM 증강 이미지를 이용한 모델 훈련**:
    *   훈련 세트의 각 이미지 샘플에 대해 증강된 버전 $(x_{\text{aug}, i}, y_i)$를 얻습니다.
    *   **옵션 1 (증강 이미지만 사용)**: 의료 영상 분할 모델 $M$을 오직 SAM 증강 이미지에 대해서만 훈련합니다.
        $$ \sum_{i=1}^{n} \text{loss}(M(x_{\text{aug}, i}), y_i) $$
    *   **옵션 2 (원본 및 증강 이미지 모두 사용)**: SAM이 신뢰할 수 없는 사전 맵을 생성할 수 있는 경우를 대비하여 원본 이미지와 SAM 증강 이미지를 모두 사용하여 모델을 훈련합니다.
        $$ \sum_{i=1}^{n} \beta\text{loss}(M(x_i), y_i) + \lambda\text{loss}(M(x_{\text{aug}, i}), y_i) $$
        (기본적으로 $\beta=1$, $\lambda=1$로 설정됩니다.)
    *   손실 함수로는 공간 교차 엔트로피 손실 또는 Dice 손실이 사용될 수 있으며, Adam [10]과 같은 SGD 기반 최적화 프로그램이 적용됩니다.

4.  **SAM 증강 이미지를 이용한 모델 배포 (테스트)**:
    *   **옵션 1 (증강 이미지만으로 훈련된 모델)**: 훈련된 모델은 SAM 증강 이미지만을 입력으로 받습니다.
        $$ \hat{y} = \tau(M(x_{\text{aug}})) $$
    *   **옵션 2 (원본 및 증강 이미지 모두로 훈련된 모델)**:
        *   **앙상블 (Ensemble)**: 원본 이미지 $x$와 SAM 증강 이미지 $x_{\text{aug}}$를 모두 사용하여 두 번 추론하고 결과를 평균 앙상블합니다.
            $$ \hat{y} = \tau(M(x) + M(x_{\text{aug}})) $$
        *   **엔트로피 기반 선택 (Entropy-based Selection)**: 두 출력 후보 중 예측 불확실성(엔트로피)이 더 낮은 것을 선택합니다.
            $$ \hat{y} = \tau(M(x^*)), \quad \text{where } x^* = \text{argmin}_{x' \in \{x, x_{\text{aug}}\}} \text{Entropy}(\tau(M(x'))) $$
            엔트로피가 낮을수록 모델의 예측 확실성이 높으며, 이는 종종 분할 정확도와 양의 상관관계를 가집니다.

## 📊 Results
SAMAug 방법의 효과는 Polyp, MoNuSeg, GlaS 세 가지 벤치마크 데이터셋에서 검증되었습니다.

*   **Polyp Segmentation (Polyp 데이터셋)**:
    *   최신 모델인 HSNet에 SAMAug를 적용하여 성능을 평가했습니다.
    *   SAMAug가 적용된 HSNet은 CVC-ClinicDB 및 CVC-ColonDB 데이터셋에서 Dice 점수를 상당히 향상시켰으며, 다른 세 데이터셋에서는 기존 HSNet과 동등한 성능을 유지했습니다.

*   **Cell Segmentation (MoNuSeg 데이터셋)**:
    *   U-Net, P-Net, Attention U-Net 모델에 SAMAug를 적용하여 세포 분할 결과를 평가했습니다.
    *   SAMAug는 U-Net, P-Net, Attention U-Net 모두에서 AJI (Aggregated Jaccard Index) 및 F-score를 크게 향상시켜 명확한 이점을 보였습니다.
    *   SAM 자체는 정확한 세포 분할을 제공하지 못했지만, 후속 딥러닝 모델이 훨씬 더 정확한 작업별 분할 결과를 생성할 수 있도록 일반적인 분할 지각 사전(perceptual prior)을 제공했습니다.

*   **Gland Segmentation (GlaS 데이터셋)**:
    *   U-Net 모델에 SAMAug를 적용하여 선(gland) 분할 결과를 평가했습니다.
    *   SAMAug가 적용된 U-Net은 SAMAug가 적용되지 않은 U-Net보다 F-score와 Object Dice 모두에서 상당히 우수한 성능을 보였습니다.

## 🧠 Insights & Discussion
SAM은 의료 영상 분할을 직접적으로 고품질로 수행하지는 못하지만, SAM이 생성하는 마스크, 특징, 안정성 점수 등은 더 나은 의료 영상 분할 모델을 구축하고 훈련하는 데 매우 유용하다는 것이 입증되었습니다. SAM은 "일반적인 분할 지각 사전"을 제공하여 특정 의료 도메인에 특화된 모델의 학습을 보조하는 역할을 합니다.

**한계점 및 향후 연구 방향**:
*   더욱 견고하고 발전된 증강 함수를 설계하는 연구가 필요합니다.
*   SAMAug 체계에서 SAM 적용의 효율성을 개선해야 합니다.
*   불확실성 추정 및 기타 임상 지향적 응용 분야에서 SAMAug를 활용하는 연구가 가능합니다.

## 📌 TL;DR
이 논문은 일반 분할 모델인 SAM이 의료 영상 분할에 직접 적용하기 어렵다는 문제에 주목하여, SAM의 출력(마스크, 안정성 점수)을 활용해 의료 영상 분할 모델의 입력 이미지를 증강하는 SAMAug 방법을 제안합니다. SAMAug는 SAM이 생성한 분할 및 경계 사전 맵을 원본 이미지 채널에 추가하여 모델 훈련을 향상시키며, 이는 U-Net, HSNet 등 다양한 CNN 및 Transformer 기반 모델의 성능을 Polyp, MoNuSeg, GlaS와 같은 의료 영상 데이터셋에서 효과적으로 개선함을 입증했습니다. 이는 SAM이 의료 영상 분야에서 직접적인 분할보다는 고수준의 "사전 지식"을 제공하여 기존 모델의 성능을 부스팅하는 데 기여할 수 있음을 보여줍니다.