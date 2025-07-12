# Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, Humphrey Shi

## 🧩 Problem to Solve

최근 사전 학습된 시각-언어 모델(예: CLIP)은 제로샷 분할(zero-shot segmentation) 태스크에 많이 활용되고 있습니다. 일반적인 접근 방식은 먼저 마스크 제안(mask proposals)을 생성한 다음 CLIP을 사용하여 이를 분류하는 "고정된 CLIP(frozen CLIP)" 패러다임을 따릅니다. 하지만 본 논문은 CLIP이 이미지 수준의 감독(image-level supervision)으로 학습되어 **다른 마스크 제안에 둔감하여 동일 이미지의 다양한 마스크 제안에 대해 유사한 예측을 하는 경향**이 있음을 밝힙니다. 이러한 둔감성은 마스크 제안 분류 시 수많은 오탐(false positives)을 초래하여, 픽셀 수준 정보(예: 배경 노이즈)에 둔감하게 만듭니다.

## ✨ Key Contributions

* **마스크 인식 미세 조정(Mask-aware Fine-tuning, MAFT) 방법 제안**: CLIP을 마스크 제안에 민감하게 만들면서도 제로샷 전이성(zero-shot transferability)을 희생하지 않는 간단하지만 효과적인 방법론을 제안합니다.
* **이미지-제안 CLIP 인코더(Image-Proposals CLIP Encoder, IP-CLIP Encoder) 설계**: 이미지와 마스크 제안을 동시에 처리할 수 있도록 CLIP 이미지 인코더를 수정하고, 마스크 제안을 멀티헤드 어텐션(Multihead Attention)의 어텐션 바이어스(attention bias)로 활용합니다.
* **마스크 인식 손실($L_{\text{ma}}$) 및 자기 증류 손실($L_{\text{dis}}$) 도입**:
  * $L_{\text{ma}}$는 마스크 제안의 IoU 점수와 IP-CLIP 인코더의 분류 점수 사이의 거리를 최소화하여 CLIP이 다양한 제안을 구별하도록 촉진합니다.
  * $L_{\text{dis}}$는 고정된 CLIP을 교사 네트워크(teacher network)로 활용하여 IP-CLIP 인코더의 출력을 정렬함으로써 CLIP의 제로샷 전이성을 유지합니다.
* **플러그 앤 플레이(Plug-and-play) 방식**: MAFT는 미세 조정 과정에서 새로운 매개변수를 도입하지 않고도 대부분의 기존 "고정된 CLIP" 기반 제로샷 분할 방법론에 쉽게 통합될 수 있습니다.
* **우수한 성능 향상**: COCO, Pascal-VOC, ADE20K 등 인기 있는 제로샷 벤치마크와 오픈-어휘(open-vocabulary) 설정에서 최신 방법들의 성능을 크게 향상시켰습니다.

## 📎 Related Works

* **초기 제로샷 분할 연구**: SPNet [31], ZS5 [1], CaGNet [10] 등은 픽셀-어휘 임베딩 공간 학습, 단어 임베딩 기반 특징 생성, 컨텍스트 정보 활용 등에 중점을 두었습니다.
* **시각-언어 모델(VLM) 기반 접근 방식**: 최근에는 CLIP [28] 및 ALIGN [17]과 같은 대규모 VLM의 풍부한 정렬 특징을 활용하는 LSeg [8], STRICT [25] 등의 연구가 등장했습니다.
* **"고정된 CLIP" 패러다임**: ZSSeg [33], OVSeg [20], FreeSeg [27], OpenSeg [9] 등은 마스크 제안을 생성한 후 고정된 CLIP으로 분류하는 방식을 채택합니다.
* **사전 학습 모델 미세 조정**: 데이터 부족 태스크(few-shot, zero-shot)에서 과적합(overfitting) 문제로 인해 어려움을 겪습니다. 이를 해결하기 위해 텍스트 또는 이미지 프롬프트 학습 [43, 42], 소수 매개변수 미세 조정 [30], 대조 학습 [38, 37] 등이 연구되었습니다. 많은 제로샷/오픈-어휘 분할 접근 방식은 전이성 유지를 위해 사전 학습 모델의 매개변수를 고정하는 것을 선호합니다.
* **MaskCLIP [41]의 한계**: MaskCLIP은 오픈-어휘 분할을 위해 CLIP을 미세 조정하려 했지만, 픽셀 수준 태스크와 이미지 수준 태스크 간의 큰 도메인 차이로 인해 실패했습니다. 본 연구는 이 관찰에서 영감을 받아 영역(region) 수준에서 CLIP을 마스크 인식적으로 미세 조정하는 데 집중합니다.

## 🛠️ Methodology

MAFT(Mask-Aware Fine-tuning)는 CLIP을 마스크 인식 표현을 학습하도록 미세 조정하는 방법입니다.

1. **이미지-제안 CLIP 인코더(IP-CLIP Encoder)**:
    * CLIP 이미지 인코더를 수정하여 임의의 수의 이미지와 마스크 제안을 동시에 처리합니다.
    * **마스크 제안을 멀티헤드 어텐션의 어텐션 바이어스(attention bias)로 적용**하여 특정 마스크 영역 내의 정보에만 집중하도록 유도합니다.
    * $L$번째 레이어부터 클래스 임베딩 벡터($F_{i}^{\text{cls}}$)를 마스크 제안 수 $N$만큼 반복하여 $F_{i*}^{\text{cls}}$를 생성합니다.
    * **$L$까지의 트랜스포머 레이어**: 표준 CLIP과 동일하게 $F_{i}^{\text{cls}}$가 $F_{i}^{\text{feat}}$의 모든 픽셀과 교차 어텐션(cross-attention)하여 컨텍스트 정보를 유지합니다.
    * **$12-L$ 트랜스포머 레이어**:
        * $F_{i*}^{\text{cls}}$ 전파: 마스크된 멀티헤드 어텐션을 사용합니다. 어텐션 바이어스 $B$는 마스크($M[n]=1$) 내부 픽셀에만 $0$을, 외부 픽셀에는 $-\infty$를 부여하여 해당 마스크 내부의 정보만 고려합니다.
        * $F_{i}^{\text{feat}}$ 전파: 표준 멀티헤드 어텐션을 사용하며 마스크에 의해 방해받지 않습니다.
    * 이를 통해 각 마스크 제안 $M[n]$에 해당하는 클래스 임베딩 $F_{i*}^{\text{cls}}[n]$이 $M[n]$ 내의 픽셀에만 어텐션을 수행합니다.

2. **목표 함수(Objective Function)**:
    * **마스크 인식 손실($L_{\text{ma}}$)**:
        * 높은 품질의 제안에는 높은 점수를, 낮은 품질의 제안에는 낮은 점수를 할당하는 것을 목표로 합니다.
        * 정답(ground-truth)과 마스크 제안 간의 IoU(Intersection over Union) 점수($S_{\text{IoU}}$)를 계산합니다.
        * $S_{\text{IoU}}$와 CLIP의 분류 점수($A_c$)의 값 범위 불일치를 해결하기 위해 $S_{\text{IoU}}$를 [0, 1] 범위로 **min-max 정규화**합니다: $S_{\text{IoU}}^{\text{norm}} = \frac{S_{\text{IoU}} - \min(S_{\text{IoU}})}{\max(S_{\text{IoU}}) - \min(S_{\text{IoU}})}$.
        * 정규화된 $S_{\text{IoU}}^{\text{norm}}$와 $A_c$ 중 선택된 클래스 간의 차이를 최소화하기 위해 **SmoothL1Loss**를 사용합니다: $L_{\text{ma}}(A_{c}^{\text{select}}, S_{\text{IoU}}^{\text{norm}}) = \text{SmoothL1}(A_{c}^{\text{select}}, S_{\text{IoU}}^{\text{norm}})$.
    * **자기 증류 손실($L_{\text{dis}}$)**:
        * CLIP의 전이성을 유지하고 학습 데이터에 대한 과적합을 완화합니다.
        * 고정된 CLIP을 교사 네트워크로, IP-CLIP을 학생 네트워크로 사용하여 자기 증류를 수행합니다.
        * 마스크를 포함하지 않는 IP-CLIP의 출력($A_S$)과 고정된 CLIP의 출력($A_T$) 간의 차이를 SmoothL1Loss로 최소화합니다: $L_{\text{dis}}(A_S, A_T) = \text{SmoothL1}(A_S, A_T)$.
    * **최종 손실 함수**: $L = L_{\text{ma}} + \lambda L_{\text{dis}}$ ($\lambda=1$).

* **효율적인 미세 조정**: 몇 번의 반복(1 epoch 미만)만으로도 효과적으로 학습됩니다.

## 📊 Results

* **제로샷 분할 성능 (미확인 클래스 mIoU, mIoU$_u$)**:
  * MAFT는 최신 방법들의 성능을 크게 향상시킵니다:
    * COCO: 50.4% (FreeSeg 대비 +8.2%p)
    * Pascal-VOC: 81.8% (FreeSeg 대비 +3.2%p)
    * ADE20K: 8.7% (FreeSeg 대비 +4.3%p)
  * 앙상블(ensemble) 전략 제거 시 MAFT의 성능 향상이 더욱 두드러집니다 (FreeSeg의 hIoU 기준 COCO +19.1%p, VOC2012 +7.0%p, ADE20K +8.3%p).
* **오픈-어휘 분할 성능**:
  * A-847, A-150, PC-459, PC-59, PAS-20 데이터셋에서 FreeSeg 대비 각각 +3.0%p, +11.2%p, +6.4%p, +19.1%p, +4.4%p 성능 향상을 달성하며, 모든 데이터셋에서 OpenSeg보다 우수한 결과를 보였습니다.
* **기여도 분석(Ablation Study)**:
  * IP-CLIP 인코더는 컨텍스트 정보를 활용하여 계산 비용을 줄이고 성능을 향상시킵니다.
  * $L_{\text{ma}}$가 주요 성능 향상을 가져오고, $L_{\text{dis}}$는 미확인 클래스에 대한 전이성을 유지하여 추가적인 성능 향상(2.6%p)에 기여합니다.
  * $L_{\text{ma}}$에는 SmoothL1Loss가 가장 효과적이며, 1k 반복 학습이 최적의 제로샷 능력을 제공합니다.
  * CLIP 내 `conv.`, `cls.`, `pos.`, `mlp` 유닛을 고정했을 때 mIoU$_u$가 5.0%p 향상되어 전체 미세 조정보다 더 나은 결과를 보였습니다.
  * 마스크 어텐션 시작 레이어 $L$을 8로 설정했을 때 가장 좋은 성능을 보였습니다.
* **SAM(Segment Anything Model) 확장**: SAM-H를 제안 생성기로 사용했을 때, SAM + MAFT는 SAM 단독보다 훨씬 우수하며, FreeSeg + MAFT보다도 뛰어난 성능을 보였습니다 (Pascal-VOC mIoU$_u$에서 6.8%p 향상).
* **다른 시각-언어 모델(VLM) 확장**: CLIP-ViT-L 및 CLIP-Res50과 같은 다양한 CLIP 백본에서도 MAFT는 일관된 성능 향상을 보이며 새로운 최첨단(state-of-the-art) 결과를 달성했습니다.

## 🧠 Insights & Discussion

* **핵심 통찰**: 기존 CLIP은 이미지 수준 학습으로 인해 픽셀/영역 수준의 세분화 태스크에서 "마스크-비인식(mask-unaware)" 문제를 겪고, 이는 오탐 및 불완전한 분할로 이어집니다.
* **MAFT의 효과**: MAFT는 IP-CLIP 인코더와 특별히 설계된 손실 함수를 통해 CLIP을 마스크 인식적으로 미세 조정함으로써 이 문제를 성공적으로 해결합니다. 이를 통해 CLIP이 마스크 제안의 품질과 내용에 따라 정확하게 반응하고, 실제 양성(true positives)이 두드러지도록 만듭니다.
* **장점**: MAFT는 효율적이며, CLIP의 원래 전이성을 보존하고, 기존 "고정된 CLIP" 접근 방식에 쉽게 적용 가능한 플러그 앤 플레이 방식입니다.
* **시사점**: 이 연구는 사전 학습된 VLM을 분할 태스크에 더 효과적으로 활용하기 위한 새로운 미세 조정 프레임워크를 제시합니다.
* **한계**: 본 MAFT는 제로샷 분할 연구에 CLIP 미세 조정 프레임워크를 도입했지만, 여전히 새로운 클래스에 대한 분류 능력은 사전 학습된 시각-언어 모델의 내재된 한계에 의해 제한됩니다. 이 한계를 더욱 좁히는 것이 향후 연구의 초점입니다. 또한, 자기 학습(Self-Training, ST) 전략의 경우 미확인 클래스를 학습 중에 얻어야 하는 요구 사항 때문에 오픈-어휘 설정과 같은 다양한 시나리오로의 일반화에 한계가 있습니다.

## 📌 TL;DR

CLIP 기반 제로샷 분할에서 CLIP의 마스크 제안에 대한 둔감성으로 인한 오탐 문제가 발생합니다. 본 논문은 이를 해결하기 위해 **마스크 인식 미세 조정(MAFT)** 방법을 제안합니다. MAFT는 **이미지-제안 CLIP 인코더(IP-CLIP Encoder)**를 통해 마스크 정보를 어텐션에 통합하고, 마스크 제안 품질에 따라 분류 점수를 조절하는 **마스크 인식 손실($L_{\text{ma}}$)**과 CLIP의 전이성을 유지하는 **자기 증류 손실($L_{\text{dis}}$)**을 사용하여 CLIP을 마스크 인식적으로 미세 조정합니다. 결과적으로 MAFT는 COCO, Pascal-VOC, ADE20K 등 다양한 제로샷 및 오픈-어휘 벤치마크에서 기존 최신 방법들의 성능을 크게 향상시키며, 마스크 영역에 민감하면서도 제로샷 능력을 유지하는 플러그 앤 플레이 솔루션을 제공합니다.
