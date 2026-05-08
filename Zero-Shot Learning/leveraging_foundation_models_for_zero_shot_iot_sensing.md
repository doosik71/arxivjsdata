# Leveraging Foundation Models for Zero-Shot IoT Sensing

Dinghao Xue, Xiaoran Fan, Tao Chen, Guohao Lan and Qun Song (2024)

## 🧩 Problem to Solve

본 논문은 IoT 센싱 데이터(mmWave, IMU, Wi-Fi 등)를 처리하는 딥러닝 모델이 학습 단계에서 보지 못한 클래스(unseen classes)를 인식하지 못하는 문제를 해결하고자 한다.

일반적인 지도 학습(Supervised Learning) 기반 모델은 훈련 데이터셋에 포함된 클래스(seen classes)에 대해서는 높은 성능을 보이지만, 새로운 클래스가 등장했을 때 이를 분류하는 능력이 부족하다. 이를 해결하기 위해 Zero-Shot Learning (ZSL)이 제안되었으나, 기존의 IoT ZSL 방식들은 다음과 같은 한계를 가진다.

1. **수동 속성 설계의 한계:** 사람이 직접 속성을 정의하는 방식은 노동 집약적이며 복잡한 데이터셋으로 확장하기 어렵다.
2. **시맨틱 갭(Semantic Gap):** Word2Vec나 BERT 같은 단어 표현 모델을 사용할 경우, IoT 태스크와 무관한 텍스트 정보(noise)가 포함되어 데이터와 임베딩 간의 괴리가 발생한다.
3. **데이터 수집의 어려움:** 이미지나 텍스트와 달리 IoT 데이터는 사람이 직관적으로 해석하기 어렵고 레이블링 비용이 매우 높아, 대규모 데이터셋을 구축하기 어렵다.

따라서 본 연구의 목표는 웹 규모의 방대한 데이터로 학습된 Foundation Model (FM)의 일반화된 지식을 활용하여, IoT 센서 신호를 텍스트 시맨틱 공간으로 정렬함으로써 보지 못한 클래스에 대해서도 높은 인식 성능을 보이는 Zero-Shot IoT Sensing 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 IoT 데이터 임베딩을 FM(특히 CLIP의 text encoder)이 생성한 시맨틱 임베딩과 정렬시키는 것이다. 이를 위해 다음과 같은 세 가지 핵심 설계를 제안한다.

1. **하이브리드 프롬프트 엔지니어링 (Hybrid Prompt Engineering):** 데이터로부터 자동으로 최적화되는 **Learnable Soft Prompt**와 LLM(GPT-3.5)을 통해 도메인 지식을 반영한 **Auxiliary Hard Prompt**를 **Cross-Attention** 메커니즘으로 결합하여, 더욱 정교한 클래스 프로토타입(Class Prototype)을 생성한다.
2. **데이터 증강을 통한 편향 제거 (Data Augmentation for Bias Mitigation):** ZSL 모델이 학습 데이터에 포함된 seen 클래스로 편향되는 문제를 해결하기 위해, GAN(Generative Adversarial Network) 기반의 생성 모델을 사용하여 unseen 클래스의 가상 데이터를 합성하고 이를 통해 모델을 미세 조정(fine-tuning)한다.
3. **엣지-클라우드 협력 구조 (Edge-Cloud Collaborative Architecture):** 엣지 디바이스에서 **Open-Set Detection**을 통해 seen/unseen 여부를 먼저 판단하고, unseen으로 판별된 데이터만 클라우드의 FM으로 전송하여 Zero-Shot 분류를 수행함으로써 효율성과 정확도를 동시에 확보한다.

## 📎 Related Works

- **Foundation Models (FMs):** CLIP과 같은 비전-언어 모델은 이미지와 텍스트를 공동 시맨틱 공간으로 정렬하여 뛰어난 zero-shot 전이 능력을 보여주었다. 최근에는 오디오, IMU 등으로 확장되고 있으나, mmWave나 Wi-Fi 같은 특수 IoT 신호에 대한 연구는 부족한 실정이다.
- **Zero-Shot Learning (ZSL):**
  - **Embedding-based:** 데이터 특징 공간을 시맨틱 공간으로 투영하는 방식이나, unseen 데이터의 부재로 인해 seen 클래스로 편향되는 경향이 있다.
  - **Generative-based:** unseen 클래스의 특징을 합성하여 지도 학습 방식으로 해결하려 하지만, 생성 모델의 학습 불안정성과 모델 붕괴(model collapse) 문제가 존재한다.
- **Zero-Shot IoT Sensing:** 기존 연구들은 수동 속성 정의나 단순 Word Embedding에 의존했으나, 이는 task-irrelevant noise가 많아 성능 향상에 한계가 있었다.

## 🛠️ Methodology

### 1. 클래스 프로토타입 추출 (Class Prototype Extraction)

각 클래스를 대표하는 시맨틱 벡터인 프로토타입을 생성하기 위해 두 가지 프롬프트를 결합한다.

- **Learnable Soft Prompt:** 클래스 토큰 주변에 학습 가능한 벡터 $\ell_i$를 배치하여 $\Phi_{text}$를 통해 텍스트 임베딩 $t_l(c)$를 추출한다. 이는 데이터 기반으로 최적화된다.
- **Auxiliary Hard Prompt:** GPT-3.5를 이용해 각 클래스를 구분 짓는 물리적 특성과 도메인 지식이 담긴 묘사 텍스트를 생성하고, 이를 통해 $t_a(c)$를 추출한다.
- **Cross-Attention 결합:** $t_a$를 Key($K$)로, $t_l$을 Query($Q$)와 Value($V$)로 설정하여 두 정보를 융합한다.
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_K}}\right), \quad t = AV$$
여기서 $t$가 최종적인 클래스 프로토타입이 된다.

### 2. IoT 임베딩 추출 (IoT Embedding Extraction)

입력된 IoT 데이터 $x_i$는 특징 추출기 $\mu(\cdot)$ (CNN, Transformer 등)를 거쳐 특징 $h_i$가 되고, 이를 임베딩 프로젝터 $g(\cdot)$를 통해 시맨틱 공간의 벡터 $e_i = g(h_i)$로 투영한다.

### 3. 모델 학습 (Model Training)

- **Supervised Contrastive Learning:** seen 클래스 데이터셋 $D_s$를 사용하여, 동일 클래스의 데이터-텍스트 쌍은 가깝게, 서로 다른 클래스는 멀게 학습시킨다. 손실 함수 $L$은 다음과 같이 정의된다.
$$L = \sum_{i \in I} \left( \frac{-1}{|P(i)+1|} \left( \sum_{p \in P(i)} \frac{e_i \cdot e_p}{\tau} + \frac{e_i \cdot t_j}{\tau} \right) + \log \left( \sum_{a \in A(i)} \exp\left(\frac{e_i \cdot e_a}{\tau}\right) + \sum_{n \in N(j)} \left(\exp\left(\frac{e_i \cdot t_n}{\tau}\right) + \exp\left(\frac{t_j \cdot t_n}{\tau}\right)\right) \right) \right)$$
여기서 $e_i$는 IoT 임베딩, $t_j$는 클래스 프로토타입, $\tau$는 temperature 파라미터이다.
- **Data Augmentation:** WGAN 기반의 생성자 $G(\cdot)$를 학습시켜 unseen 클래스의 가상 IoT 데이터 $\tilde{x}$를 생성한다. 이 데이터를 사용하여 특징 추출기와 프로젝터를 미세 조정함으로써 seen 클래스로의 편향을 줄인다.

### 4. Zero-Shot 분류 절차 (Zero-Shot Classification)

1. **Open-Set Detection (Edge):** 입력 데이터 $e_{test}$와 seen 클래스 클러스터들 사이의 유클리드 거리를 계산한다. 특정 임계값 $\lambda_i$보다 가까운 샘플이 없으면 'Unseen'으로 판별한다.
2. **Zero-Shot Learning (Cloud):** 'Unseen'으로 판별된 데이터는 클라우드로 전송되어, unseen 클래스 프로토타입 $\{t(c^u_i)\}$와의 내적(dot product) 값이 가장 높은 클래스로 분류된다.
$$\hat{y}_{det} = \text{argmax}_{c^u_i \in U} (e_{det} \cdot t(c^u_i)^T)$$

## 📊 Results

### 실험 설정

- **데이터셋:** USC-HAD, PAMAP2 (IMU), MM-Fi (mmWave, Wi-Fi).
- **평가 지표:**
  - Open-Set Detection: Precision, Recall, F1 score.
  - GZSL: $\text{ACC}_S$ (seen 정확도), $\text{ACC}_U$ (unseen 정확도), $\text{ACC}_H$ (두 정확도의 조화 평균).

### 정량적 결과

- **Open-Set Detection:** 모든 데이터셋에서 제안 방법이 MSP, KNN, MCM 등 베이스라인보다 높은 F1 score를 기록하였다. 특히 IMU 데이터(USC-HAD)에서 F1 score 78.8%를 달성하며 우수성을 보였다.
- **Generalized ZSL:** 조화 평균 $\text{ACC}_H$ 지표에서 타 모델들을 압도하였다. 예를 들어 PAMAP2 데이터셋에서 BERT(59.3%)나 FREE(52.1%)보다 높은 62.1%의 $\text{ACC}_H$를 기록하였다. 이는 $\text{ACC}_U$의 비약적인 상승에 기인한다.

### Ablation Study 결과

- **Prompt Engineering 제거 시:** $\text{ACC}_U$와 $\text{ACC}_H$가 크게 하락하여, 하이브리드 프롬프트가 정확한 시맨틱 정렬에 필수적임을 확인하였다.
- **Open-Set Detection 제거 시:** $\text{ACC}_S$는 상승하지만 $\text{ACC}_U$가 급격히 하락한다. 이는 detection 단계가 seen 클래스로의 편향(bias)을 막아주는 결정적인 역할을 함을 의미한다.
- **Data Augmentation 제거 시:** $\text{ACC}_U$가 하락하며, 합성 데이터가 unseen 클래스 임베딩의 강건성을 높여줌을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순히 Foundation Model을 적용하는 것에 그치지 않고, IoT 센싱 데이터의 특성을 고려한 세 가지 장치(하이브리드 프롬프트, 가상 데이터 생성, Open-set detection)를 유기적으로 결합하였다는 점에서 강점이 있다.

특히, **Open-Set Detection**의 중요성이 매우 크다는 점이 인상적이다. GZSL에서 흔히 발생하는 "seen 클래스로의 쏠림 현상"을 단순히 손실 함수로 해결하려 하지 않고, 추론 단계에서 명시적으로 분리하는 파이프라인을 구축함으로써 $\text{ACC}_U$를 실질적으로 끌어올렸다.

다만, 다음과 같은 한계와 논의점이 존재한다.

1. **GAN의 의존성:** unseen 데이터를 생성하기 위해 GAN을 사용하는데, GAN 자체가 학습이 불안정하다는 특성이 있어 데이터셋의 규모나 특성에 따라 합성 데이터의 품질이 달라질 수 있다.
2. **LLM 프롬프트 의존성:** GPT-3.5가 생성한 묘사 텍스트가 도메인 지식을 충분히 반영하지 못할 경우, 하드 프롬프트의 효과가 반감될 수 있다.
3. **실시간성:** 엣지-클라우드 협력 구조를 취하고 있으나, unseen 데이터 발생 시 클라우드 통신 비용 및 지연 시간이 발생하므로 이에 대한 실시간성 분석이 추가될 필요가 있다.

## 📌 TL;DR

본 연구는 FM의 일반화된 지식을 IoT 센싱에 도입하여, 학습하지 않은 클래스를 인식하는 **Zero-Shot IoT Sensing** 프레임워크를 제안한다. 핵심은 **(1) 도메인 지식(Hard)과 학습 가능 벡터(Soft)를 결합한 프롬프트 엔지니어링**, **(2) GAN 기반 가상 데이터 생성을 통한 편향 제거**, **(3) 엣지에서의 Open-set detection을 통한 효율적 분류**이다. 이 방식은 IMU, mmWave, Wi-Fi 등 다양한 IoT 모달리티에서 기존 ZSL 방식보다 뛰어난 일반화 성능을 보였으며, 향후 다양한 센서 환경으로 확장될 가능성이 높다.
