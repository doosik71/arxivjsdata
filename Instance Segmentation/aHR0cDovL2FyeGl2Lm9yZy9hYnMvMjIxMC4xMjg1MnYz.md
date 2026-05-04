# 1st Place Solution of The Robust Vision Challenge 2022 Semantic Segmentation Track

Junfei Xiao, Zhichao Xu, Shiyi Lan, Zhiding Yu, Alan Yuille, Anima Anandkumar (2022)

## 🧩 Problem to Solve

본 논문은 실세계의 다양한 환경("in the wild")에서 딥러닝 모델이 겪는 취약성, 특히 데이터 분포의 변화(distributional shifts)와 자연적인 섭동(natural perturbations)으로 인한 성능 저하 문제를 해결하고자 한다. 일반적인 시맨틱 세그멘테이션(Semantic Segmentation) 모델은 특정 데이터셋에 과적합되는 경향이 있어, 학습하지 않은 새로운 도메인의 데이터에 적용했을 때 강건성(Robustness)이 크게 떨어진다.

이 연구의 목표는 ECCV 2022에서 개최된 Robust Vision Challenge (RVC)의 시맨틱 세그멘테이션 트랙에서 우승하는 것으로, 단일 모델이 실내/실외, 합성/실제 이미지 등 서로 다른 특성을 가진 6개의 벤치마크 데이터셋 모두에서 높은 일반화 성능을 유지하도록 하는 것이다.

## ✨ Key Contributions

본 솔루션의 핵심 아이디어는 **강건함이 검증된 Vision Transformer(ViT) 기반의 백본**과 **대규모 다중 데이터셋 학습 전략**을 결합하는 것이다. 구체적으로는 다음과 같은 설계를 통해 강건성을 확보하였다.

1.  **FAN-B-Hybrid 백본 채택**: 이미지 분류 및 다운스트림 태스크에서 뛰어난 정확도와 강건성을 보인 Fully Attentional Network (FAN)를 인코더로 사용하여 Out-of-Distribution (OOD) 시나리오에 대한 대응력을 높였다.
2.  **SegFormer 프레임워크 활용**: 효율적인 MLP 디코더를 통해 다중 레벨 특징을 융합하는 SegFormer 구조를 채택하여 세그멘테이션 성능을 극대화하였다.
3.  **광범위한 데이터셋 통합 및 균형화**: 9개의 서로 다른 데이터셋을 통합하여 학습 데이터의 다양성을 확보하고, 데이터셋 간 크기 차이로 인한 불균형을 해소하기 위해 단순하지만 효과적인 데이터 리사이징(Resizing) 전략을 사용하였다.

## 📎 Related Works

논문에서는 도메인 일반화(Domain Generalization)를 위한 기존의 접근 방식들로 도메인 랜덤화(Domain Randomization), 도메인 불변 표현 학습(Domain Invariant Representation Learning), disentanglement 학습 및 메타 학습(Meta Learning) 등을 언급한다. 특히, 여러 데이터셋을 조합하고 라벨 공간을 정렬하여 학습시키는 multi-dataset training 방식이 강력한 도메인 일반화 방법론보다 더 나은 성능을 낼 수 있음을 시사하는 기존 연구([10])를 참고하였다.

또한, 최근의 Vision Transformer (ViT) 연구들이 CNN 기반 모델보다 OOD 시나리오에서 더 강건한 특성을 보인다는 점에 주목하였다. 예를 들어, SegFormer는 자연적인 오염이 포함된 Cityscapes-C 데이터셋에서 CNN 기반 방법론보다 월등한 성능을 보였으며, FAN은 분류 및 세그멘테이션 작업 모두에서 최신 수준의 강건성을 입증하였다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 **FAN-B-Hybrid**를 인코더(Encoder)로, **SegFormer**를 전체적인 세그멘테이션 프레임워크로 사용하는 구조이다. 

1.  **Encoder (Backbone)**: FAN-B-Hybrid를 사용하며, ImageNet-22K에서 사전 학습된 가중치로 초기화한 후 ImageNet-1K에서 미세 조정(Fine-tuning)된 체크포인트를 사용한다.
2.  **Decoder**: SegFormer의 MLP 기반 디코더를 사용하여 초기 Convolution 블록, 마지막 FAN Transformer 블록, 그리고 최종 Class Attention 블록의 출력을 융합하여 최종 세그멘테이션 마스크를 예측한다.

### 학습 전략 및 절차
-   **학습 데이터 구성**: ADE20K, Cityscapes, Mapillary Vistas, ScanNet, VIPER, WildDash 2, IDD, BDD, COCO 등 총 9개의 데이터셋을 통합하여 사용한다.
-   **데이터 균형화 (Dataset Balancing)**: 데이터셋마다 이미지 수가 매우 다르므로(예: COCO는 WildDash 2보다 30배 이상 큼), 각 데이터셋을 $\text{120,000 // len(dataset)}$ 번 반복하여 샘플링하는 단순 리사이징 전략을 통해 데이터 불균형 문제를 완화하였다.
-   **통합 라벨 공간 (Unified Label Space)**: 서로 다른 데이터셋의 라벨을 256개의 클래스로 구성된 통합 라벨 공간으로 투영(Project)하여 학습하였다.
-   **손실 함수**: 표준적인 교차 엔트로피 손실(Cross-Entropy Loss)을 사용하였으며, 별도의 손실 가중치 조정이나 하이퍼파라미터 튜닝은 진행하지 않았다.
-   **추론 및 후처리**: 예측된 세그멘테이션 맵은 통합 라벨 공간에서 각 데이터셋의 원래 라벨 공간으로 다시 투영하는 후처리 과정을 거친다.

### 구현 세부 사항
-   **최적화**: AdamW 옵티마이저를 사용하였으며, 학습률(Learning rate)은 $6 \times 10^{-5}$, Weight decay는 $0.01$로 설정하였다.
-   **스케줄러**: Poly 학습률 스케줄러를 적용하였으며, 1,500 iteration의 Warmup 기간을 두었다.
-   **학습 설정**: 총 80,000 iteration 동안 학습하였으며, 초기 절반의 학습 단계에서는 BDD와 IDD 데이터셋을 제외하였다.
-   **컴퓨팅 자원**: 64장의 NVIDIA V100 (32G) GPU를 사용하였으며, 총 학습 시간은 약 35시간이 소요되었다.

## 📊 Results

본 솔루션은 RVC 2020의 우승 솔루션들(MSEG, SNRN)과 비교하여 6개의 모든 벤치마크 데이터셋에서 압도적인 성능 향상을 보였다. 성능 지표로는 class mIoU가 사용되었다.

| Method | ADE20K | Cityscapes | Mapillary | ScanNet | VIPER | WildDash 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MSEG (2020) | 33.1 | 80.7 | 34.1 | 48.5 | 40.7 | 34.7 |
| SNRN (2020) | 31.1 | 74.7 | 40.4 | 54.6 | 62.5 | 42.2 |
| **FAN (Ours)** | **43.4** | **82.0** | **55.2** | **58.6** | **69.8** | **47.5** |

결과적으로 본 방법론은 모든 도메인에서 기존의 최선 방법론들을 큰 격차로 제치고 1위를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구의 결과는 **강력한 ViT 백본(FAN)**과 **방대한 양의 다중 도메인 데이터 학습**이 결합되었을 때, 시맨틱 세그멘테이션 작업에서 매우 높은 일반화 능력과 강건성을 확보할 수 있음을 입증한다. 특히 복잡한 하이퍼파라미터 튜닝이나 정교한 손실 함수 설계 없이도 아키텍처의 선택과 데이터 구성만으로 성능을 극대화했다는 점이 인상적이다.

### 한계 및 논의사항
논문에서는 ViT 모델의 실제 적용 시 다음과 같은 한계점을 명시하고 있다.
1.  **자원 소모**: 데이터셋의 규모와 라벨 공간이 커짐에 따라 학습 시 요구되는 계산량과 메모리 소비가 급격히 증가하는 문제가 발생한다.
2.  **추론 효율성**: 현재의 ViT 모델들은 연산 복잡도가 높아 실제 디바이스에 배포하여 실시간으로 적용하는 데 있어 효율성 문제가 여전히 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

본 논문은 FAN-B-Hybrid 백본과 SegFormer 프레임워크를 결합하고, 9개의 대규모 데이터셋을 통합 학습시킨 전략을 통해 RVC 2022 시맨틱 세그멘테이션 트랙에서 1위를 차지한 솔루션을 제안한다. 이는 ViT의 내재적인 강건성과 다중 데이터셋 학습의 시너지 효과를 보여주며, 향후 다양한 도메인에 적용 가능한 강건한 세그멘테이션 모델 구축을 위한 강력한 베이스라인을 제시한다.