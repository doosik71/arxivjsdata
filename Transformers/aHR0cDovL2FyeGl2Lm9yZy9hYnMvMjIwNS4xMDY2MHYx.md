# VISION TRANSFORMERS IN 2022: AN UPDATE ON TINY IMAGENET
Ethan M. Huynh

## 🧩 해결하고자 하는 문제
최근 Vision Transformer(ViT) 모델들은 이미지 분류 작업에서 뛰어난 성능을 보이며 기존 CNN 아키텍처와의 격차를 크게 좁혔습니다. 하지만 이러한 모델들은 주로 ImageNet-21k와 같은 대규모 데이터셋으로 사전 학습된 후 ImageNet-1k나 CIFAR-10/100과 같은 데이터셋으로 파인튜닝되는 방식으로 성능이 평가되어 왔습니다.

이 과정에서 ImageNet-1k의 하위 집합이면서도 200개 클래스의 100,000개 이미지로 구성된 중요한 소규모 데이터셋인 Tiny ImageNet에 대한 최신 Vision Transformer들의 전이 학습(transfer learning) 성능 평가는 간과되었습니다. 기존 Tiny ImageNet 관련 연구는 대부분 스크래치(scratch) 학습에 초점을 맞추었으며, 전이 학습의 효과를 제대로 다루지 않았습니다. 이 논문은 이러한 연구 공백을 메우고자 합니다.

## ✨ 주요 기여
*   ViT, DeiT, CaiT, Swin Transformer 등 최신 Vision Transformer 모델들의 Tiny ImageNet 데이터셋에 대한 전이 학습 성능을 체계적으로 평가하고 보고합니다.
*   Swin Transformer가 Tiny ImageNet에서 91.35%의 검증 정확도를 달성하며 새로운 SoTA(State-of-the-Art) 성능을 수립했음을 보여줍니다. 이는 기존 최고 기록을 0.33% 능가하는 결과입니다.
*   다양한 데이터 증강(data augmentation) 및 정규화(regularization) 기법에 대한 어블레이션 스터디(ablation study)를 통해 Tiny ImageNet에 최적화된 훈련 설정을 탐색합니다.
*   모델의 파라미터 수(parameter count)와 FLOPs(floating-point operations)가 반드시 훈련 효율성(throughput)을 나타내지 않음을 보여주고, 레이어 수(layer count)가 효율성에 더 큰 영향을 미칠 수 있음을 시사합니다.

## 📎 관련 연구
*   **Vision Transformer (ViT)** (Dosovitskiy et al., 2020): 트랜스포머를 이미지 분류에 적용할 수 있음을 처음으로 보인 연구. 대규모 데이터셋(JFT-300M) 사전 학습의 필요성을 제기.
*   **Data-efficient Image Transformer (DeiT)** (Touvron et al., 2020): 엄격한 훈련 스케줄과 지식 증류(knowledge distillation)를 통해 대규모 데이터의 필요성을 완화하고 ImageNet-21k 및 ImageNet-1k에서 ViT 훈련 가능성을 제시.
*   **Class Attention in Image Transformer (CaiT)** (Touvron et al., 2021): DeiT의 설계를 따르며 ViT의 성능 향상을 목표로 함.
*   **Swin Transformer** (Liu et al., 2021b): Shifted window를 사용한 계층적 트랜스포머 구조로, 다양한 비전 태스크에서 뛰어난 성능을 보임.
*   **Tiny ImageNet에 대한 이전 연구** (Lee et al., 2021): Tiny ImageNet에서 Vision Transformer를 스크래치부터 훈련하여 정확도를 개선하는 연구를 수행했으나, 전이 학습에 대한 평가는 부족했음.

## 🛠️ 방법론
이 연구는 `timm` 라이브러리(Wightman, 2019)에서 제공하는 Vision Transformer 모델들을 사용하고 DeiT와 유사한 훈련 레지멘트를 적용합니다.

1.  **모델 선택**: ViT-L/16, DeiT-B/16-D (지식 증류 버전), CaiT-S/36, Swin-L/4 모델을 평가에 사용했습니다.
2.  **하드웨어**: Nvidia RTX 3070 (8GB 메모리) 및 8코어 CPU를 사용하여 훈련했습니다.
3.  **데이터 처리**:
    *   입력 이미지 해상도: 384x384 (훈련 및 테스트 모두)
    *   데이터 증강: Mixup (확률 0.8), Cutmix (확률 1.0), Random Erasing (확률 0.25)을 적용했습니다. (RandAugment는 최종 설정에서 제외됨)
4.  **훈련 설정**:
    *   옵티마이저: AdamW
    *   초기 학습률: $10^{-3}$ (코사인 감쇠 스케줄 사용)
    *   가중치 감쇠(weight decay): 0.05
    *   에포크 수: 30
    *   배치 크기: 128 (8GB GPU 메모리 제약으로 인해 그레디언트 누적(gradient accumulation) 사용)
    *   정규화: 레이블 스무딩(label smoothing) ($\alpha=0.1$), Stochastic Depth (0.1)

5.  **훈련 절차 튜닝**:
    *   다양한 학습률 및 가중치 감쇠 조합으로 AdamW를 튜닝했습니다.
    *   SAM, ASAM, PUGD와 같은 교란된 옵티마이저(perturbed optimizers) 및 SGD도 테스트했지만, AdamW가 초기 정확도와 훈련 시간 면에서 가장 우수했습니다.
    *   데이터 증강 및 정규화 기법에 대한 어블레이션 스터디를 수행하여 최적의 조합을 찾았습니다. RandAugment, AutoAugment, Random Resized Crop, Simple Random Crop, Model EMA 등의 효과를 분석했습니다.

## 📊 결과
*   **정확도 결과 (Tiny ImageNet 검증)**:
    *   Swin-L/4: **91.35%** (새로운 SoTA)
    *   DeiT-B/16-D: 87.29%
    *   CaiT-S/36: 86.74%
    *   ViT-L/16: 86.43%

*   **효율성 분석**:
    *   DeiT-B/16-D는 가장 빠르게 훈련되었으며, 지식 증류의 효과를 입증했습니다.
    *   Swin-L/4는 가장 높은 정확도를 달성했습니다.
    *   CaiT-S/36은 파라미터 수와 FLOPs가 가장 적음에도 불구하고, 가장 낮은 처리량(throughput)을 보이며 훈련 시간이 가장 길었습니다. 이는 파라미터 수와 FLOPs가 항상 모델 효율성을 직접적으로 나타내지는 않음을 시사합니다.
    *   모델 효율성은 레이어 수 및 임베딩 크기와 더 밀접한 관련이 있는 것으로 나타났습니다. (예: DeiT는 12개 레이어, CaiT는 36개 레이어)

*   **어블레이션 스터디 결과**:
    *   기본 설정에서 RandAugment를 제거하면 정확도가 소폭 상승하거나 유지되면서 모델이 최고 정확도를 더 오래 유지하는 안정적인 훈련 패턴을 보였습니다.
    *   Mixup, CutMix, Random Erasing, Stochastic Depth, Label Smoothing은 모두 학습에 유익했습니다.
    *   AutoAugment, Random Resized Crop, Simple Random Crop, Model EMA는 성능 향상에 도움이 되지 않거나 오히려 정확도를 감소시켰습니다.

## 🧠 통찰 및 논의
*   **Tiny ImageNet에 대한 전이 학습의 효과**: 이 연구는 대규모 데이터셋으로 사전 학습된 Vision Transformer가 Tiny ImageNet과 같은 소규모 데이터셋으로도 성공적으로 전이 학습될 수 있음을 명확히 보여줍니다. 이는 Tiny ImageNet이 ImageNet-1k의 하위 집합이라는 점을 고려할 때 합리적인 결과입니다.
*   **Swin Transformer의 우수성**: Swin Transformer는 윈도우 기반의 Multi-Headed Self-Attention (MSA)과 Shifted window를 활용하는 독특한 아키텍처 덕분에 Tiny ImageNet에서도 탁월한 성능을 발휘하여 Vision Transformer의 선두 주자임을 다시 한번 입증했습니다.
*   **DeiT의 효율성**: 지식 증류를 적용한 DeiT는 합리적인 정확도를 유지하면서도 가장 빠른 훈련 속도를 보여주어, 데이터 효율적인 모델의 가치를 강조합니다.
*   **성능 지표의 한계**: 파라미터 수와 FLOPs만으로는 모델의 실제 훈련 효율성(처리량)을 완전히 예측하기 어렵다는 통찰을 제공합니다. 특히 CaiT의 사례에서 볼 수 있듯이, 레이어 수와 임베딩 크기 같은 다른 요인들이 훈련 속도에 더 큰 영향을 미칠 수 있습니다.
*   **최적의 훈련 설정**: 다양한 데이터 증강 및 정규화 기법의 효과를 분석하여 Tiny ImageNet에 대한 최적의 훈련 설정(AdamW, Mixup, CutMix, Random Erasing, Stochastic Depth, Label Smoothing)을 제안합니다. 특히 RandAugment의 제거가 오히려 훈련 안정성에 긍정적인 영향을 미친다는 점이 주목할 만합니다.
*   **향후 연구**: SwinV2 (Liu et al., 2021a) 및 MiniViT (Zhang et al., 2022)와 같이 더 진보된 Vision Transformer 아키텍처에 대한 Tiny ImageNet 성능 평가가 향후 연구로 제안됩니다.

## 📌 요약
이 논문은 ImageNet의 소규모 부분집합인 Tiny ImageNet에 대한 최신 Vision Transformer(ViT, DeiT, CaiT, Swin)의 전이 학습 성능을 업데이트합니다. 연구 결과, Swin Transformer가 91.35%의 정확도로 Tiny ImageNet에서 새로운 SoTA를 달성했습니다. DeiT는 효율성 면에서 뛰어났습니다. 또한, 파라미터 수와 FLOPs가 반드시 훈련 효율성을 나타내지는 않으며, 레이어 수가 더 중요한 요소임을 발견했습니다. 데이터 증강 및 정규화 기법에 대한 어블레이션 스터디를 통해 Tiny ImageNet에 최적화된 훈련 설정을 제시했습니다.