# CPNet: Cycle Prototype Network for Weakly-supervised 3D Renal Compartments Segmentation on CT Images

Song Wang, Yuting He, Youyong Kong, Xiaomei Zhu, Shaobo Zhang, Pengfei Shao, Jean-Louis Dillenseger, Jean-Louis Coatrieux, Shuo Li, and Guanyu Yang (2021)

## 🧩 Problem to Solve

본 논문은 복부 CT 영상에서 신장 구획(Renal Compartments), 즉 신피질(Renal Cortex)과 신수질(Renal Medulla)의 3D 구조를 분할하는 문제를 다룬다. 신장 구획의 정확한 분할은 부분 신절제술(Laparoscopic Partial Nephrectomy) 시 신장 기능 손실을 최소화하고 수술 성공률을 높이는 데 매우 중요하다.

그러나 CT 영상 기반의 신장 구획 분할은 다음과 같은 세 가지 주요 난제가 존재한다.

1. **불분명한 경계(Unclear Boundaries):** 신피질, 신수질 및 신장 외부 조직 간의 CT 수치(Hounsfield Unit)가 매우 유사하여, 모델이 구별 가능한 특징을 추출하기 어렵고 과분할(Over-segmentation) 또는 과소분할(Under-segmentation)이 발생하기 쉽다.
2. **얇은 구조(Thin Structures):** 신피질이 신장 내부로 뻗어 나가 신수질과 얽혀 있으며, 이로 인해 매우 얇고 불안정한 형태의 구조가 형성된다. 수용 영역(Receptive Field)이 큰 일반적인 특징 추출기는 이러한 세밀한 특징(Fine-grained features)을 손실하기 쉽다.
3. **해부학적 변이 및 데이터 부족(Anatomy Variation & Small Dataset):** 환자마다 신수질의 모양과 개수가 매우 다양하다. 정교한 어노테이션(Annotation)에는 전문의의 많은 시간이 소요되므로 라벨링된 데이터셋의 규모가 작으며, 이는 모델의 일반화 성능을 저하시키는 원인이 된다.

따라서 본 논문의 목표는 적은 수의 라벨만을 사용하여도 높은 정확도로 신장 구획을 분할할 수 있는 자동화된 라벨 효율적(Label-efficient) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 위에서 언급한 문제들을 해결하기 위해 **CPNet(Cycle Prototype Network)**을 제안하며, 핵심 아이디어는 다음과 같다.

1. **Cycle Prototype Learning (CPL):** 순방향(Forward) 및 역방향(Reverse) 프로세스를 통해 일관성(Consistency)을 학습함으로써 모델의 일반화 능력을 높이고 라벨 노이즈에 대한 강건함을 확보한다.
2. **Bayes Weakly Supervised Module (BWSM):** 서로 다른 시점의 CT 영상(CTA 및 CTU)에서 나타나는 조영제 반응의 차이라는 사전 지식(Prior knowledge)과 네트워크의 예측값(Likelihood)을 베이즈 정리(Bayes theory)로 결합하여 정확한 의사 라벨(Pseudo-label)을 생성한다.
3. **Fine Decoding Feature Extractor (FDFE):** 인코더-디코더 구조에서 전역적 형태 정보(Global morphology)와 국소적 세부 정보(Local detail)를 결합하여, 얇은 구조물에 대해 정교한 분할이 가능하도록 세밀한 특징 맵을 추출한다.

## 📎 Related Works

기존의 신장 구획 분할 연구는 크게 두 가지 방향으로 진행되었다.

- **반자동 방식(Semi-automatic methods):** 수동 어노테이션과 이미지 연산을 결합하여 분할을 수행하지만, 많은 노동력이 소요된다는 한계가 있다.
- **딥러닝 기반 자동 방식:** 자동으로 분할을 수행하지만, 라벨링된 데이터셋의 규모가 매우 작아 성능이 심각하게 제한되는 문제가 있다.

본 논문은 특히 Few-shot segmentation 분야의 **PANet**과 같은 Prototype 기반 방법론을 참고하였으나, 단순한 프로토타입 정렬을 넘어 Cycle 구조의 학습과 베이즈 기반의 의사 라벨링, 그리고 세밀한 특징 추출기를 도입함으로써 기존의 한계를 극복하고자 하였다.

## 🛠️ Methodology

CPNet의 전체 시스템은 크게 CPL, BWSM, FDFE 세 가지 모듈로 구성된다.

### 1. Cycle Prototype Learning (CPL)

CPL은 support 이미지와 query 이미지 간의 특징 일관성을 학습하여 일반화 성능을 높이는 구조이다.

- **순방향 프로세스 (Forward Process):**
  - Support 이미지 $i_s$와 query 이미지 $i_q$로부터 특징 $x_s, x_q$를 추출한다.
  - Support 라벨 $y_s$를 이용해 Masked Average Pooling $M(\cdot)$을 수행하여 서포트 프로토타입 $t_s$를 생성한다: $t_s = M(y_s \cdot x_s)$.
  - query 특징 $x_q$와 $t_s$ 사이의 코사인 유사도(Cosine Similarity, $CS(\cdot)$)를 계산하여 query 예측값 $y'_q$를 얻는다: $y'_q = CS(t_s \cdot x_q)$.
- **역방향 프로세스 (Reverse Process):**
  - 순방향에서 얻은 $y'_q$를 라벨로 사용하여 query 프로토타입 $t_q$를 생성한다: $t_q = M(y'_q \cdot x_q)$.
  - 이 $t_q$를 이용해 다시 support 이미지 $x_s$를 예측하여 $y'_s$를 얻는다: $y'_s = CS(t_q \cdot x_s)$.

- **손실 함수 (Loss Function):**
    순방향의 query 손실 $L_{query}$(예측값 $y'_q$와 의사 라벨 $\hat{y}_q$ 사이의 교차 엔트로피)와 역방향의 support 손실 $L_{support}$(예측값 $y'_s$와 실제 라벨 $y_s$ 사이의 교차 엔트로피)를 결합하여 전체 손실을 정의한다.
    $$L_{total} = \theta L_{query} + L_{support}$$
    여기서 $\theta$는 두 손실의 균형을 맞추는 하이퍼파라미터이다.

### 2. Bayes Weakly Supervised Module (BWSM)

BWSM은 라벨이 없는 데이터에서 사전 지식을 추출하여 고품질의 의사 라벨을 생성한다.

- **사전 지식 추출:** CTA(동맥기)와 CTU(배설기) 영상의 조영제 반응 차이를 이용하여 사전 특징 맵 $f_q$를 생성하고, 이를 통해 정답일 확률 $p_{correct}$와 오답일 확률 $p_{wrong}$을 계산한다.
- **베이즈 보정:** 네트워크의 소프트맥스(Softmax) 출력 확률을 가능도(Likelihood, $l_{correct}, l_{wrong}$)로 사용하여, 사후 확률(Posterior probability) 기반의 의사 라벨을 생성한다.
  - 보정 과정의 수식은 다음과 같다.
    $$\begin{cases} p_{correct} = 1/3 + \omega \\ p_{wrong} = (1 - p_{correct})/2 \\ f_{correct} = \frac{p_{correct} \cdot l_{correct}}{p_{correct} \cdot l_{correct} + p_{wrong} \cdot l_{wrong}} \\ f_{wrong} = \frac{p_{wrong} \cdot l_{wrong}}{p_{correct} \cdot l_{correct} + p_{wrong} \cdot l_{wrong}} \end{cases}$$
    여기서 $\omega$는 사전 확률의 영향력을 조절하는 하이퍼파라미터이다.

### 3. Fine Decoding Feature Extractor (FDFE)

FDFE는 얇은 신장 구조를 정밀하게 포착하기 위해 설계되었다.

- **구조:** 인코더는 $3\times3\times3$ 컨볼루션 레이어 2개와 $2\times2\times2$ 풀링 레이어로 구성된 블록의 반복으로 이루어지며, 각 레이어 뒤에는 Group Norm이 적용된다.
- **핵심 기제:** 디코더의 업풀링(Up-pooling)을 통해 전역적 형태 정보를 복원하는 동시에, 인코더와 디코더 사이의 스킵 연결(Skip Connection)을 통해 다운샘플링 과정에서 손실된 국소적 세부 정보를 직접 전달한다. 이를 통해 최종 특징 맵의 경계를 더욱 날카롭게(Sharp) 만들어 얇은 구조의 분할 성능을 높인다.

## 📊 Results

### 실험 설정

- **데이터셋:** 복부 강화 CT 영상에서 추출한 60개의 신장 ROI (크기 $160\times160\times200$).
- **학습/테스트 분할:** 훈련 세트 30개, 테스트 세트 30개.
- **약지도 학습 조건:** 훈련 세트 중 단 **4장의 이미지**에만 정밀 라벨을 부여하여 학습을 진행하였다.
- **평가 지표:** Dice-M/C (수질/피질), Average Hausdorff Distance (AHD-M/C).

### 주요 결과

- **정량적 결과:** CPNet-Total 모델은 단 4장의 라벨만으로 피질(Cortex) Dice 78.4%, 수질(Medulla) Dice 79.1%, 평균 Dice 78.7%를 달성하였다.
- **비교 분석:**
  - **SegNet, U-Net:** 매우 적은 라벨 데이터로는 충분한 지식을 학습하지 못해 신장 전체를 수질로 판단하는 등 제대로 된 분할을 수행하지 못했다.
  - **PANet (Prototype 기반):** 평균 Dice 56.3%를 기록하였으며, 대략적인 분할은 가능하나 세부 구조의 손실이 심했다.
  - **CPNet:** PANet 대비 약 20% 이상의 Dice 성능 향상을 보였으며, 시각적으로도 훨씬 정교한 분할 결과를 나타냈다.

### 모듈별 기여도 분석 (Ablation Study)

- **FDFE의 영향:** VGG16과 같은 일반적인 특징 추출기를 사용했을 때보다 FDFE를 사용했을 때 평균 Dice가 약 20% 향상되었다. 이는 얇은 구조 분할에 FDFE가 핵심적임을 시사한다.
- **BWSM의 영향:** BWSM을 적용하지 않았을 때보다 평균 Dice가 약 2% 향상되어, 의사 라벨을 통한 데이터 증강 효과가 입증되었다.

## 🧠 Insights & Discussion

본 논문은 극소량의 라벨링 데이터만으로도 고성능의 3D 의료 영상 분할이 가능함을 보여주었다. 특히 다음의 통찰을 얻을 수 있다.

1. **일관성 학습의 중요성:** CPL의 역방향 프로세스는 일종의 정규화(Regularization) 역할을 하여, 모델이 라벨의 노이즈에 덜 민감하게 만들고 일반화 능력을 향상시킨다.
2. **도메인 특화 사전 지식의 활용:** 단순히 데이터량을 늘리는 것이 아니라, CTA/CTU 영상의 조영제 반응 차이라는 의학적 사전 지식을 베이즈 정리를 통해 결합함으로써, 단순한 네트워크 예측보다 훨씬 정확한 의사 라벨을 생성할 수 있었다.
3. **구조적 정밀도의 필요성:** 의료 영상, 특히 신장 구획처럼 구조가 얇은 경우 전역적 정보와 국소적 정보를 동시에 보존하는 FDFE와 같은 특화된 디코더 구조가 필수적이다.

**한계점 및 논의:**

- $\omega$ 하이퍼파라미터에 따라 성능 변화가 나타나는데, 이는 사전 지식과 네트워크 예측 사이의 균형점을 찾는 것이 매우 중요함을 의미한다.
- 4장의 매우 적은 라벨을 사용했음에도 성능이 높지만, 라벨 수가 증가함에 따라 성능이 점차 안정화(Stabilize)되는 양상을 보인다. 이는 데이터가 극도로 부족한 상황에서의 효율성은 입증되었으나, 충분한 데이터가 있을 때의 상한선에 대해서는 추가 연구가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 단 4장의 라벨링된 3D CT 영상만으로 신장 구획(피질, 수질)을 정교하게 분할하는 **CPNet**을 제안한다. **CPL**을 통한 일관성 학습, **BWSM**을 이용한 베이즈 기반 의사 라벨 생성, **FDFE**를 통한 세밀한 특징 추출을 결합하여, 기존의 프로토타입 기반 모델(PANet) 대비 Dice 성능을 약 20% 이상 향상시켰다. 이 연구는 어노테이션 비용이 매우 높은 의료 영상 분야에서 약지도 학습(Weakly-supervised learning)의 실용적인 적용 가능성을 보여주었다.
