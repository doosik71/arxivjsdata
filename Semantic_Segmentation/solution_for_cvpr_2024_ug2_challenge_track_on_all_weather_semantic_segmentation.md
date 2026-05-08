# Solution for CVPR 2024 UG2+ Challenge Track on All Weather Semantic Segmentation

Jun Yu, Yunxiang Zhang, Fengzhao Sun, Leilei Wang, Renjie Lu (2024)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 악천후(Adverse Weather) 상황에서 발생하는 시각적 저하로 인해 시맨틱 세그멘테이션(Semantic Segmentation)의 성능이 급격히 저하되는 현상을 극복하는 것이다. 자율 주행, 로보틱스, 장면 이해와 같은 분야에서 세그멘테이션은 매우 중요한 기술이지만, 기존의 벤치마크 데이터셋(예: ADE20K, Cityscapes)은 주로 맑은 날씨의 표준적인 환경을 다루고 있어 실제 세계의 복잡한 기상 조건에서는 모델의 강건성(Robustness)이 떨어진다는 문제가 있다.

특히 기존의 악천후 연구들은 합성 데이터(Synthetic data)에 의존하거나, 저하된 이미지와 깨끗한 이미지 사이의 정렬(Alignment)이 맞지 않는 데이터셋을 사용하여 학습 효율이 낮다는 한계가 있었다. 따라서 본 논문의 목표는 실제 세계의 다양한 기상 조건에서도 정확하고 강건한 세그멘테이션 결과를 도출할 수 있는 최적의 파이프라인을 구축하여 CVPR 2024 UG2+ 챌린지에서 높은 성능을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 설계 아이디어는 대규모 기초 모델(Foundation Model)의 강력한 표현 능력과 정교한 데이터 증강, 그리고 모델 앙상블 전략을 결합하는 것이다. 주요 기여 사항은 다음과 같다.

- **대규모 기초 모델의 도입**: Deformable Convolution을 핵심 연산자로 사용하는 InternImage-H를 백본으로 채택하여, ViT(Vision Transformer)에 필적하는 확장성과 적응적 공간 집계(Adaptive spatial aggregation) 능력을 확보하였다.
- **다단계 데이터 증강 전략**: 학습 데이터셋의 한계를 극복하기 위해 오프라인(Offline)과 온라인(Online) 증강 기법을 동시에 적용하여 모델의 일반화 성능을 높였다.
- **모델 퓨전 및 최적화**: 단일 모델의 한계를 넘기 위해 여러 체크포인트 모델의 결과를 하드 보팅(Hard Voting) 방식으로 결합하는 앙상블 전략을 통해 최종 정확도와 강건성을 향상시켰다.

## 📎 Related Works

논문에서는 자율 주행 및 장면 이해를 위한 시맨틱 세그멘테이션의 중요성을 언급하며, ADE20K와 Cityscapes 같은 표준 데이터셋에서의 성과를 소개한다. 그러나 이러한 데이터셋들은 악천후 상황을 충분히 반영하지 못한다는 한계가 있다.

기존의 악천후 대응 연구들은 주로 합성 데이터를 생성하여 학습시키는 방식을 사용했으나, 이는 실제 환경과의 괴리가 존재하며 정렬 문제(Misalignment)를 야기한다. 본 연구는 이러한 한계를 극복하기 위해 고품질의 정렬된 쌍(Paired) 데이터를 제공하는 WeatherProof 데이터셋을 활용하며, 이는 기존의 합성 기반 접근 방식보다 더 정확한 평가와 학습을 가능하게 한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구의 파이프라인은 `InternImage-H` 인코더와 `UperNet` 디코더로 구성된 구조를 기반으로 한다. 전체적인 흐름은 사전 학습된 가중치를 로드한 후, WeatherProof 데이터셋에 맞게 미세 조정(Fine-tuning)하고, 최종적으로 여러 모델의 결과물을 앙상블 하는 과정을 거친다.

### 주요 구성 요소

1. **Encoder: InternImage-H**
   InternImage는 전통적인 CNN의 엄격한 유도 편향(Inductive bias)을 줄이고 ViT처럼 대규모 파라미터를 학습할 수 있도록 설계된 모델이다. 핵심 연산자인 $\text{DCNv3 (Deformable Convolution v3)}$를 통해 입력 이미지와 태스크 정보에 따라 수용 영역(Receptive field)을 적응적으로 조절한다. 이를 통해 객체의 형태나 크기가 다양한 악천후 환경에서도 강력한 객체 표현 능력을 갖춘다.

2. **Decoder: UperNet**
   UperNet은 $\text{Pyramid Pooling Module (PPM)}$과 $\text{Feature Pyramid Network (FPN)}$의 아이디어를 통합한 구조이다. 다양한 스케일의 특징 맵(Feature map)을 융합하여 서로 다른 크기의 객체를 효과적으로 세그멘테이션한다. 본 연구에서는 InternImage의 출력층에 $\text{Layer Normalization (LN)}$, $\text{Feed-Forward Network (FFN)}$, 그리고 $\text{GELU}$ 활성화 함수를 결합하여 성능을 보완하였다.

### 학습 절차 및 최적화

- **초기화**: ADE20K 데이터셋으로 사전 학습된 InternImage-H 가중치를 사용하였다.
- **최적화 알고리즘**: $\text{AdamW}$ 옵티마이저를 사용하였으며, 설정값은 다음과 같다.
  - 초기 학습률(Learning rate): $0.00002$
  - 가중치 감쇠(Weight decay): $0.05$
  - 크롭 사이즈(Crop size): $960$
- **하드웨어**: 8대의 NVIDIA Tesla V100 GPU를 사용하여 총 6,000번의 반복 학습(Iterations)을 수행하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: WeatherProof 데이터셋을 사용하였다. 이 데이터셋은 174,000장 이상의 이미지로 구성되어 있으며, 동일한 장면의 맑은 날씨 이미지와 악천후 이미지 쌍(Pair)으로 이루어져 있어 제어된 환경에서의 정확한 평가가 가능하다. (학습 세트: 513세트 $\times$ 300쌍, 검증 세트: 38세트)
- **평가 지표**: $\text{mIoU (mean Intersection over Union)}$를 주요 지표로 사용하였다.

### 데이터 증강 결과

학습 데이터의 부족을 해결하기 위해 다음과 같은 기법을 적용하였다.

- **Offline Augmentation**: 이미지의 대비(Contrast)와 밝기(Brightness)를 수정하여 데이터셋의 양을 물리적으로 확장하였다.
- **Online Augmentation**: 학습 과정에서 $\text{RandomCrop}$, $\text{RandomFlip}$, $\text{Padding}$을 적용하여 모델의 과적합을 방지하고 강건성을 높였다.

### 정량적 결과

검증 세트 분석 결과, 3,500 iteration 시점에서 최적의 성능을 보이는 모델을 확인하였다. 이후 최적 모델과 인접한 성능의 모델들을 대상으로 하드 보팅(Hard Voting) 앙상블을 수행한 결과, 테스트 세트에서 최종적으로 $0.4371$의 mIoU를 달성하였다.

| 모델/방법 | 테스트 mIoU |
| :--- | :---: |
| iter 3000 | 0.4040 |
| iter 3500 | 0.4198 |
| iter 4000 | 0.4184 |
| **Voting results** | **0.4371** |

## 🧠 Insights & Discussion

본 연구의 강점은 최신 대규모 기초 모델인 InternImage를 세그멘테이션 태스크에 성공적으로 적용하고, 이를 UperNet과 결합하여 악천후라는 특수한 환경에서도 높은 성능을 이끌어냈다는 점이다. 특히, 단일 모델에 의존하지 않고 여러 체크포인트를 앙상블하는 전략이 mIoU를 유의미하게 상승시켰음을 알 수 있다.

다만, 본 보고서는 챌린지 결과 리포트 형식으로 작성되어 다음과 같은 한계가 존재한다.
첫째, 사용된 데이터 증강 기법(밝기, 대비 조정 등)이 구체적으로 어떤 범위로 적용되었는지에 대한 상세 수치가 명시되지 않았다.
둘째, InternImage-H 외에 다른 백본 모델과의 비교 실험(Ablation Study)이 부족하여, InternImage가 구체적으로 어떤 부분에서 악천후 상황에 더 유리했는지에 대한 분석이 부족하다.
셋째, 하드 보팅 외에 소프트 보팅이나 가중 평균 등 다른 앙상블 기법과의 비교 분석이 이루어지지 않았다.

그럼에도 불구하고, 대규모 모델과 정교한 데이터 증강의 조합이 실제 세계의 시각적 저하 문제를 해결하는 데 있어 효과적인 전략임을 입증하였다.

## 📌 TL;DR

본 논문은 CVPR 2024 UG2+ 챌린지를 위해 $\text{InternImage-H}$ 백본과 $\text{UperNet}$ 디코더를 결합한 악천후 시맨틱 세그멘테이션 솔루션을 제안한다. 오프라인/온라인 데이터 증강과 다중 모델 하드 보팅 앙상블을 통해 테스트 세트에서 $0.4371$ mIoU를 기록하며 챌린지 3위를 차지하였다. 이 연구는 대규모 기초 모델의 적응적 수용 영역 능력이 기상 악화로 인한 이미지 저하 상황에서도 강건한 성능을 낼 수 있음을 시사하며, 향후 자율 주행 시스템의 안전성 향상을 위한 모델 설계 방향을 제시한다.
