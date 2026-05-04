# CROSS-LEVEL CONTRASTIVE LEARNING AND CONSISTENCY CONSTRAINT FOR SEMI-SUPERVISED MEDICAL IMAGE SEGMENTATION

Xinkai Zhao, Chaowei Fang, De-Jun Fan, Xutao Lin, Feng Gao, Guanbin Li (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 데이터 어노테이션에 소요되는 막대한 비용과 노동력을 줄이기 위해 준지도 학습(Semi-Supervised Learning, SSL)을 적용하는 문제를 다룬다.

의료 영상 전문가들의 경험에 따르면, 병변이나 폴립(polyp)과 같은 대상 객체를 식별하는 데 있어 텍스처(texture), 광택(luster), 매끄러움(smoothness)과 같은 국소적 특성(local attributes)이 매우 중요한 요소이다. 그러나 일반적인 CNN 기반 모델에서는 층이 깊어질수록 이러한 국소적 특징들이 점차 희석되는 경향이 있다.

기존의 대조 학습(Contrastive Learning, CL) 방식들은 이미지 레벨(image-wise), 패치 레벨(patch-wise), 또는 포인트 레벨(point-wise) 등 동일한 수준의 특징들만을 비교하는 한계가 있었다. 따라서 본 논문의 목표는 전역적(global) 표현과 국소적(local) 패치 표현 사이의 관계를 탐구하는 **Cross-level** 접근 방식을 통해, 제한된 레이블 데이터만으로도 국소 특징에 대한 표현 능력을 강화하여 분할 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 전체에서 얻은 전역적 정보와 이미지의 일부 패치에서 얻은 국소적 정보 사이의 상관관계를 학습시키는 것이다. 이를 위해 다음과 같은 두 가지 핵심 장치를 제안한다.

1. **Cross-level Contrastive Learning**: 전역 특징 맵(global feature map)의 특정 지점과, 해당 지점을 포함하는 국소 패치 특징 맵(local feature map)의 대응 지점을 긍정 쌍(positive pair)으로 설정하여 대조 학습을 수행함으로써 국소 특징의 표현력을 높인다.
2. **Cross-level Consistency Constraint**: 특징 레벨뿐만 아니라 최종 예측(prediction) 레벨에서도 전역 이미지의 예측 결과와 국소 패치의 예측 결과가 일치하도록 강제하여, 시맨틱(semantic) 차원에서의 일관성을 확보한다.

## 📎 Related Works

논문은 준지도 학습을 위한 기존의 접근 방식들을 다음과 같이 설명한다.

* **Self-labeling**: 모델이 스스로 생성한 의사 레이블(pseudo labels)을 사용하여 학습하는 방식이다. 하지만 의사 레이블의 품질에 의존적이며, 잘못된 예측을 스스로 강화하는 확증 편향(confirmation bias) 문제가 발생할 수 있다.
* **Self-supervised Learning**: 상대적 위치 예측이나 회전 예측과 같은 pretext task를 통해 표현력을 높이는 방식이다. 최근에는 SimCLR와 같은 대조 학습(Contrastive Learning)이 주목받고 있다.
* **기존 Contrastive Learning의 한계**: 기존의 CL 방식들은 이미지 레벨의 임베딩만을 비교하여 공간적 관계 정보를 무시하거나, 동일한 레벨(포인트-포인트 등) 내에서만 비교를 수행한다. 전역 레벨과 국소 레벨을 교차하여 비교하는 Cross-level similarity cue에 대한 탐구는 부족한 상태였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 U-Net을 백본(backbone)으로 사용하며, 전역 이미지와 국소 패치를 동시에 입력으로 받아 특징을 추출한다. 전체 손실 함수는 지도 학습 손실($L_{sup}$), 전역-국소 대조 손실($L_{contrast}$), 전역-국소 일관성 손실($L_{consist}$)의 합으로 구성된다.

### 1. Dense Cross-level Contrastive Learning

입력 이미지 $X \in \mathbb{R}^{H \times W \times 3}$가 주어졌을 때, 백본 $f$를 통해 전역 특징 맵 $Z = f(X)$를 추출한다. 동시에 이미지를 $n \times n$개의 패치 $x_i$로 분할하여 각각의 국소 특징 맵 $z_i = f(x_i)$를 추출한다. 이후 세 개의 합성곱 층으로 구성된 프로젝션 헤드(projection head) $p$를 통해 특징 벡터를 투영한다.

전역 특징 맵의 투영 결과 $p(Z)$를 $n \times n$ 블록으로 나누어 $p(Z)_i$라고 할 때, 동일한 위치의 국소 패치 특징 $p(z_i)$를 positive pair로, 전역 특징 맵 내의 다른 위치의 점들을 negative samples로 설정한다. 대조 손실 함수는 다음과 같다.

$$L_{contrast}^{(i)} = -\log \frac{\exp [p(Z)_i \cdot p(z_i) / \tau]}{\exp [p(Z)_i \cdot p(z_i) / \tau] + \sum_{n \neq i} \exp [p(Z)_i \cdot p(Z)_n / \tau]}$$

여기서 $\tau$는 온도 하이퍼파라미터이다.

### 2. Dense Patch-image Consistency Learning

특징 레벨의 유사성을 최종 예측 결과까지 확장하기 위해, 예측 헤드 $h$( $1 \times 1$ convolution)를 통해 얻은 전역 예측 결과와 국소 패치 예측 결과 사이의 MSE(Mean Square Error)를 계산한다.

$$L_{consist} = \sum_i \text{MSE}(h(z_i), h(Z)_i)$$

여기서 $h(Z)_i$는 전역 예측 결과에서 $z_i$와 동일한 위치에 해당하는 패치 영역을 의미한다.

### 3. 학습 절차 및 전체 손실 함수

전체 손실 함수는 다음과 같이 정의된다.

$$L_{all} = L_{sup} + \alpha L_{contrast} + \beta L_{consist}$$

여기서 $L_{sup}$는 Dice loss와 Cross Entropy loss의 조합으로 구성된다. 학습은 총 300 에포크(epoch) 동안 두 단계로 진행된다.

* **1단계 (100 epochs)**: $\alpha=1, \beta=0$으로 설정하여 지도 학습과 대조 학습을 통해 일반화된 표현 능력을 학습시킨다.
* **2단계 (200 epochs)**: $\alpha=0, \beta=1$로 설정하여 일관성 손실을 통해 국소 정보의 캡처 능력을 강화한다.

## 📊 Results

### 실험 설정

* **데이터셋**: Kvasir-SEG(폴립 분할, 레이블 120장/비레이블 480장), ISIC 2018(피부 병변 분할, 레이블 156장/비레이블 1400장).
* **비교 대상**: U-Net(Baseline), UAMT, URPC.
* **평가 지표**: MAE, Dice coefficient, mIoU.

### 정량적 결과

Kvasir-SEG 데이터셋에서 본 제안 방법(ours all)은 mIoU 63.50%를 기록하며 U-Net(57.08%), URPC(61.50%)보다 우수한 성능을 보였다. ISIC 2018 데이터셋에서도 mIoU 74.00%를 달성하여 타 모델(UAMT 72.16%, URPC 71.83%) 대비 성능 향상을 입증하였다.

### 절제 실험(Ablation Study)

Kvasir-SEG 결과에 따르면, $L_{contrast}$만 사용하거나 $L_{consist}$만 사용했을 때보다 두 손실 함수를 모두 사용했을 때 성능이 가장 안정적이고 높게 나타났다. 이는 특징 레벨의 대조 학습과 시맨틱 레벨의 일관성 제약이 서로 보완적인 관계임을 시사한다.

### 정성적 결과

시각화 결과, 기존 방법들은 배경이 복잡하거나 객체와 배경의 색상이 유사한 경우 분할에 실패하는 경향이 있었으나, 제안 방법은 이러한 복잡한 케이스에서도 객체 영역을 강건하게 분리해내는 모습을 보였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상의 특성인 '국소적 디테일의 중요성'을 딥러닝 아키텍처의 학습 전략으로 잘 녹여내었다. 특히 단순히 데이터 증강을 통한 일관성을 찾는 것이 아니라, **전역-국소 간의 계층적 관계(Cross-level relationship)**를 강제함으로써 네트워크가 국소 특징을 유지하도록 유도한 점이 강점이다.

다만, 학습 과정에서 $\alpha$와 $\beta$를 순차적으로 적용하는 2단계 학습 전략을 사용하였는데, 이는 학습의 안정성을 위한 선택으로 보이나, 두 손실 함수를 동시에 최적화했을 때의 영향이나 최적의 전환 시점에 대한 분석이 부족하다는 점이 아쉽다. 또한, 패치 분할 방식($n \times n$)이 고정되어 있는데, 패치 크기의 변화가 성능에 미치는 영향에 대한 고찰이 추가된다면 더 설득력 있는 연구가 되었을 것이다.

## 📌 TL;DR

본 연구는 준지도 학습 기반의 의료 영상 분할을 위해 **전역 특징과 국소 패치 특징을 교차 비교하는 대조 학습(Cross-level Contrastive Learning)**과 **예측 레벨의 일관성 제약(Consistency Constraint)**을 제안하였다. 이를 통해 데이터 레이블이 부족한 상황에서도 국소적 특징(텍스처 등)을 효과적으로 학습하여 폴립 및 피부 병변 분할 작업에서 SOTA 성능을 달성하였다. 이 접근법은 국소 특징이 중요한 다양한 의료 영상 분석 작업으로 확장될 가능성이 높다.
