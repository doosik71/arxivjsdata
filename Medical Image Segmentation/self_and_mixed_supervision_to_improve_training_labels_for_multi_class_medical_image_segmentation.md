# Self and Mixed Supervision to Improve Training Labels for Multi-Class Medical Image Segmentation

Jianfei Liu, Christopher Parnell, Ronald M. Summers (2023/2024*)

*\*원문 내 출판 연도가 명시되지 않았으나, 인용 문헌 [20]이 2023년 논문이며 본 연구가 그 후속 연구임을 고려하여 작성한다.*

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 다중 클래스(multi-class) 의료 영상 분할(segmentation)을 위한 고품질 학습 라벨의 부족이다. 의료 영상의 라벨링 작업은 도메인 전문가의 지식이 필수적이므로 비용과 시간이 매우 많이 소요된다.

특히, 기존의 자동 분할 방법들을 통해 생성된 라벨들은 대량으로 확보할 수 있으나, 정확도가 떨어지는 '약한 라벨(weak labels)'이라는 한계가 있다. 예를 들어, 복부 CT 영상에서 피하 지방(subcutaneous adipose tissue)이 내장 지방(visceral adipose tissue)으로 잘못 분류되거나 골반 근육(pelvis muscle) 영역이 누락되는 등의 문제가 빈번하게 발생한다. 따라서 본 논문의 목표는 소량의 정확한 '강한 라벨(strong labels)'과 대량의 부정확한 '약한 라벨'을 효율적으로 결합하여, 학습 과정에서 자동으로 약한 라벨의 품질을 개선하고 최종적으로 높은 분할 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Dual-branch Network**와 **Transfer Learning**을 결합하여 학습 라벨을 순차적으로 개선하는 전략이다.

단순히 강한 라벨과 약한 라벨을 한 번에 사용하여 학습시키는 것이 아니라, 다음의 단계적 접근 방식을 제안한다.

1. **Self-supervision 단계**: 약한 라벨만을 사용하여 인코더의 특징 표현(feature representation)을 먼저 학습시킨다.
2. **Mixed Supervision 단계**: 학습된 인코더를 고정(freeze)하고, 강한 라벨과 약한 라벨을 동시에 사용하여 두 개의 디코더를 미세 조정(fine-tuning)한다.
3. **Iterative Label Update**: 미세 조정 과정에서 강한 디코더(strong decoder)가 예측한 결과물을 다시 약한 라벨로 대체함으로써, 학습 데이터 자체의 정확도를 반복적으로 향상시킨다.

## 📎 Related Works

논문에서는 라벨 부족 문제를 해결하기 위한 세 가지 기존 접근 방식을 소개한다.

1. **Unsupervised Segmentation**: 오토인코더의 잠재 특징을 클러스터링하거나 데이터 정제 전략을 사용하여 라벨 없이 분할을 수행한다. 그러나 배경과의 대비가 낮은 저대비 경계(low-contrast boundaries) 구조에서는 성능이 떨어진다는 한계가 있다.
2. **Self-supervised Learning**: pretext task(예: 루빅스 큐브 게임 형태의 볼륨 변환 예측)를 통해 유용한 특징을 먼저 학습한 후 downstream task인 분할 작업으로 전이하는 방식을 사용한다.
3. **Semi-supervised Segmentation**: 소량의 라벨링된 데이터와 대량의 라벨링되지 않은 데이터를 함께 사용한다. Entropy minimization, Mean Teacher, Cross-consistency training 등이 활용되며, 최근에는 공유 인코더와 두 개의 디코더를 가진 Dual-branch 구조가 주목받고 있다.

본 연구는 이전 연구(Liu et al., 2023)에서 제안한 Dual-branch 구조를 확장하여, 다중 클래스 분할에 적용하고 **Transfer Learning**을 통해 과적합(over-fitting) 문제를 해결하고 라벨 개선 효율을 높였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 구조 (Architecture)

본 네트워크는 공유 인코더(Shared Encoder)와 두 개의 동일한 디코더(Identical Decoders)로 구성된 Dual-branch 구조를 가진다.

- **Shared Encoder**: ResNet을 백본으로 사용하며, 입력 영상에서 공통적인 특징을 추출한다.
- **Strong Decoder**: 수동 주석(manual annotation)으로 생성된 고정밀 '강한 라벨'을 처리한다.
- **Weak Decoder**: 자동 분할 방법(level-set 및 U-Net)으로 생성된 '약한 라벨'을 처리한다.

### 2. 학습 절차 (Training Pipeline)

학습은 크게 두 단계로 진행된다.

**단계 1: Self-supervised Pre-training**

- **목적**: 다중 클래스 데이터의 불균형으로 인한 과적합을 방지하고 인코더의 특징 추출 능력을 최적화한다.
- **절차**: 약한 라벨만을 사용하여 100 epoch 동안 학습한다.
- **손실 함수**: 불균형한 세그멘테이션 성능을 개선하기 위해 Generalized Dice overlap을 사용한다.
  $$ \text{Generalized Dice overlap} $$
  (상세 수식은 본문에 명시되지 않았으나, 클래스별 가중치를 적용한 Dice 계수임을 시사한다.)

**단계 2: Mixed Supervision & Fine-tuning**

- **절차**:
  1. 전 단계에서 학습된 인코더의 가중치를 고정(freeze)한다.
  2. 강한 라벨과 약한 라벨을 각각의 디코더에 입력하여 미세 조정을 수행한다.
  3. 이 과정에서 **Supervised data term**, **Cross-consistent training term**, **Prediction confidence term**을 모두 사용하여 학습을 가이드한다.
  4. **Iterative Update**: 학습 도중 강한 디코더에서 생성된 예측 마스크(segmentation mask)가 기존의 부정확한 약한 라벨을 대체하도록 업데이트한다.

### 3. 추론 및 라벨 개선 프로세스

강한 라벨로부터 학습된 지식이 강한 디코더를 통해 전달되고, 이 결과물이 다시 약한 라벨의 자리를 대체함으로써, 초기에는 누락되었던 근육 영역이나 잘못 분류된 지방 조직 영역이 점진적으로 복구된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 101명의 환자로부터 얻은 복부 CT 스캔.
- **라벨 구성**:
  - 강한 라벨: 총 155 슬라이스 중 100 슬라이스를 학습에 사용 (영상의학과 전문의 감독 하에 수동 주석).
  - 약한 라벨: 101명 전체 스캔에서 추출한 11,326개의 비주석 슬라이스.
  - 검증 데이터: 나머지 55 슬라이스.
- **측정 지표**: Intersection over Union (IoU), Dice Similarity Coefficient (DSC), Relative Volume Difference (RVD).
- **비교 대상**: 초기 약한 라벨(Initial Weak labels), Transfer learning이 적용되지 않은 Dual-branch 네트워크.

### 2. 정량적 결과

Transfer learning을 적용한 제안 방법이 모든 지표에서 가장 우수한 성능을 보였다. 특히 DSC의 향상이 두드러진다.

| 대상 조직 | 초기 약한 라벨 (%) | Dual-branch (w/o Transfer) (%) | 제안 방법 (w/ Transfer) (%) |
| :--- | :---: | :---: | :---: |
| **Muscle (근육)** | 74.2 | 88.0 | **91.5** |
| **Subcutaneous Adipose (피하 지방)** | 91.2 | 94.7 | **95.6** |
| **Visceral Adipose (내장 지방)** | 77.6 | 83.2 | **88.5** |

모든 결과는 $p < 0.05$ 수준에서 통계적으로 유의미한 향상을 보였으며, 특히 근육 조직에서 가장 큰 폭의 성능 향상이 관찰되었다.

### 3. 정성적 결과

시각적 분석 결과, 초기 약한 라벨에서 내장 지방으로 오분류되었던 피하 지방 영역이 제안 방법을 통해 정확하게 교정되었으며, 완전히 누락되었던 골반 근육 영역이 반복적인 학습 과정을 통해 완전히 복구되는 것이 확인되었다.

## 🧠 Insights & Discussion

본 연구는 의료 영상 분야에서 라벨 부족 문제를 해결하기 위해 **'약한 라벨 $\rightarrow$ 인코더 특징 학습 $\rightarrow$ 강한 라벨을 통한 디코더 교정 $\rightarrow$ 약한 라벨 업데이트'**라는 전략적인 파이프라인을 구축하였다.

**강점:**

- **데이터 효율성**: 매우 적은 양의 수동 주석(강한 라벨)만으로도 대량의 데이터셋 품질을 자동으로 높일 수 있음을 증명하였다.
- **안정성**: 인코더를 먼저 학습시키고 고정한 뒤 디코더를 튜닝하는 Transfer learning 방식을 도입하여, 클래스 불균형으로 인한 과적합 문제를 효과적으로 억제하였다.

**한계 및 논의:**

- 본 논문에서는 구체적인 손실 함수의 수학적 수식을 완전히 명시하지 않고 이전 연구([20])를 인용하고 있어, 수식 기반의 엄밀한 재현에는 제약이 있을 수 있다.
- 강한 라벨의 양이 극단적으로 적을 때(예: 100 슬라이스 미만)의 성능 하한선에 대한 분석이 부족하다.
- 또한, 약한 라벨을 생성하는 초기 알고리즘(level-set, U-Net)의 성능이 너무 낮을 경우, 초기 self-supervision 단계에서 인코더가 잘못된 특징을 학습할 위험이 있는지에 대한 고찰이 필요하다.

## 📌 TL;DR

본 논문은 소량의 정밀 라벨(Strong)과 대량의 부정확한 라벨(Weak)을 결합한 **Dual-branch Network**를 통해 의료 영상의 학습 라벨을 자동으로 개선하는 방법을 제안한다. 특히 **Self-supervision**으로 인코더를 초기화하고 **Transfer Learning**을 통해 디코더를 미세 조정하며 약한 라벨을 반복적으로 업데이트하는 전략을 사용하여, 복부 CT의 근육 및 지방 조직 분할 정확도(DSC)를 최대 91.5%까지 끌어올렸다. 이 연구는 고비용의 의료 데이터 라벨링 부담을 획기적으로 줄이면서도 고성능의 분할 모델을 구축할 수 있는 실용적인 방법론을 제시한다.
