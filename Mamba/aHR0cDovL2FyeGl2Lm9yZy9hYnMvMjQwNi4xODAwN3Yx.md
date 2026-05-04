# DEEP MAMBA MULTI-MODAL LEARNING

Jian Zhu, Xin Zou, Yu Cui, Zhangmin Huang, Chenshu Hu, Bo Lyu (2024)

## 🧩 Problem to Solve

본 논문은 서로 다른 특성을 가진 이종 데이터(Heterogeneous data)인 멀티모달 특징들을 효과적으로 융합하여 멀티미디어 검색(Multimedia Retrieval)의 성능을 높이는 문제를 해결하고자 한다. 

멀티미디어 검색에서는 이미지와 텍스트 등 다양한 모달리티의 데이터를 통합하여 효율적인 검색이 가능하도록 하는 것이 중요하다. 특히, 대규모 데이터셋에서 검색 속도를 높이기 위해 데이터를 이진 코드(Binary hash code)로 변환하는 Hashing 기술이 필수적이다. 본 연구의 목표는 Mamba 네트워크의 강력한 표현 능력을 활용하여 멀티모달 특징 융합 성능을 극대화하고, 이를 통해 높은 정확도와 빠른 추론 속도를 동시에 달성하는 Deep Mamba Multi-modal Hashing (DMMH) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba 네트워크와 CNN을 결합하여 멀티모달 특징의 정교한 의미 추출과 깊은 융합을 달성하는 것이다.

1. **Mamba 네트워크의 최초 적용**: 멀티미디어 검색 분야에 Mamba 네트워크를 처음으로 도입하여 단일 모달리티의 시맨틱 특징을 강화하였다.
2. **DMML (Deep Mamba Multi-modal Learning) 제안**: 이종 데이터의 융합을 위해 Mamba 네트워크를 통한 특징 강화 $\rightarrow$ 가산(Additive) 방식의 단순 융합 $\rightarrow$ CNN을 통한 심층 융합으로 이어지는 새로운 학습 프레임워크를 설계하였다.
3. **DMMH (Deep Mamba Multi-modal Hashing) 구현**: DMML을 기반으로 멀티모달 데이터를 효율적인 해시 코드로 변환하는 DMMH 방법론을 제안하여, 기존의 최신 기술(State-of-the-art) 대비 우수한 검색 성능을 입증하였다.

## 📎 Related Works

본 논문은 멀티뷰 해싱(Multi-view Hashing) 및 멀티모달 해싱 분야의 기존 연구들을 비교 대상으로 삼고 있다. 

- **비지도 학습 기반 방법**: Multiple Feature Hashing (MFH), Multi-view Alignment Hashing (MAH), Multi-view Latent Hashing (MVLH), Multi-view Discrete Hashing (MvDH) 등이 포함된다.
- **지도 학습 기반 방법**: Multiple Feature Kernel Hashing (MFKH)부터 최근의 Bit-aware Semantic Transformer Hashing (BSTH) 및 Deep Metric Multi-View Hashing (DMMVH)까지 총 9가지의 최신 방법론을 비교하였다.

논문에서는 구체적인 기존 연구의 이론적 한계를 상세히 서술하고 있지는 않으나, 실험 결과를 통해 기존의 Transformer 기반이나 일반적인 딥러닝 기반 해싱 방법들이 DMMH에 비해 $\text{mAP}$ (mean Average Precision) 측면에서 낮은 성능을 보임을 제시하며 차별성을 강조한다.

## 🛠️ Methodology

본 논문에서 제안하는 DMMH의 전체 파이프라인은 시각 및 텍스트 데이터를 입력받아 최종적으로 이진 해시 코드를 생성하는 구조이다. 시스템의 구성 요소와 절차는 다음과 같다.

### 1. 특징 추출 및 정규화 (Backbones & MLPs)
- **Vision Backbone**: $\text{VGGNet}$을 사용하여 이미지로부터 시각적 특징을 추출한다.
- **Text Backbone**: $\text{BoW (Bag-of-Words)}$ 모델을 사용하여 텍스트로부터 텍스트 특징을 생성한다.
- **MLPs**: 추출된 서로 다른 차원의 멀티모달 특징들을 정규화하고 차원을 일치시키기 위해 $\text{MLP}$ 네트워크를 사용한다.

### 2. 시퀀스 변환 및 의미 추출 (Dilation & Mamba Network)
- **Dilation**: Mamba 네트워크는 시퀀스 데이터를 입력으로 받으므로, $\text{Dilation}$ 네트워크를 통해 단일 모달 특징을 확장하여 시퀀스 임베딩 형태로 변환한다.
- **Mamba Network**: 변환된 시퀀스 임베딩으로부터 세밀한 시맨틱 마이닝(Fine-grained semantic mining)을 수행하여 각 모달리티의 표현력을 강화한다.

### 3. 특성 융합 및 해싱 (CNNs & Hash Layer)
- **특징 융합**: 먼저 가산(Additive) 방식을 통해 단순 융합을 수행한 후, $\text{CNN}$을 사용하여 멀티모달 특징 간의 심층 융합(Deep fusion)을 진행함으로써 시맨틱 표현 능력을 더욱 향상시킨다.
- **Hash Layer**: 최종적으로 학습된 임베딩을 이진 해시 코드로 변환하여 효율적인 검색이 가능하도록 한다.

본 논문에는 구체적인 손실 함수(Loss function)나 학습 알고리즘에 대한 수식은 명시되어 있지 않으나, 전체적인 흐름은 $\text{특징 추출} \rightarrow \text{차원 맞춤} \rightarrow \text{시퀀스 확장} \rightarrow \text{Mamba 처리} \rightarrow \text{CNN 융합} \rightarrow \text{해싱}$ 순으로 진행된다.

## 📊 Results

### 실험 설정
- **데이터셋**: $\text{MIR-Flickr25K}$, $\text{NUS-WIDE}$, $\text{MS COCO}$ 세 가지 공개 데이터셋을 사용하였다.
- **평가 지표**: 검색 성능을 측정하기 위해 $\text{mAP (mean Average Precision)}$를 사용하였다.
- **비교 대상**: 비지도 학습 4종 및 지도 학습 9종을 포함한 총 13개의 멀티뷰 해싱 방법론과 비교하였다.
- **해시 비트 수**: $16, 32, 64, 128$ bits 설정에서 성능을 측정하였다.

### 주요 결과
실험 결과, 제안된 $\text{DMMH}$가 모든 데이터셋과 모든 비트 설정에서 가장 높은 $\text{mAP}$를 기록하며 $\text{SOTA}$ 성능을 달성하였다.

- **성능 향상 폭**: 기존 최신 방법론인 $\text{DMMVH}$와 비교했을 때, 평균 $\text{mAP}$가 다음과 같이 향상되었다.
    - $\text{MIR-Flickr25K}$: $2.00\%$ 향상
    - $\text{NUS-WIDE}$: $1.43\%$ 향상
    - $\text{MS COCO}$: $4.80\%$ 향상

이러한 결과는 Mamba 네트워크의 뛰어난 표현 능력과 $\text{DMML}$의 효율적인 이종 데이터 융합 능력이 결합되어 나타난 성과로 분석된다.

## 🧠 Insights & Discussion

### 강점
본 연구는 최근 주목받는 Mamba 네트워크를 멀티모달 융합에 성공적으로 적용하여, 기존 Transformer 계열이나 CNN 단독 모델보다 뛰어난 성능을 보였다. 특히 시각 정보와 텍스트 정보를 각각 Mamba로 강화한 뒤 CNN으로 심층 융합하는 단계적 구조가 효과적이었음을 시사한다.

### 한계 및 미해결 질문
- **수학적 근거 부족**: 논문 텍스트 내에 모델의 학습 목표(Objective function)나 구체적인 손실 함수에 대한 수식이 전혀 제시되지 않아, 모델이 구체적으로 어떻게 최적화되었는지 알 수 없다.
- **추론 속도 분석**: 초록에서 "추론 속도의 이점"을 언급하였으나, 정량적인 추론 시간(Inference time)이나 복잡도(Complexity)에 대한 실험 결과 표가 제공되지 않아 실제로 얼마나 빠른지 검증하기 어렵다.
- **Mamba의 역할**: Mamba가 구체적으로 어떤 시퀀스 특성을 포착하여 성능을 높였는지에 대한 분석적 설명이 부족하다.

## 📌 TL;DR

본 논문은 Mamba 네트워크와 CNN을 결합한 멀티모달 학습 프레임워크인 $\text{DMML}$과 이를 활용한 해싱 방법론 $\text{DMMH}$를 제안한다. Mamba를 통해 단일 모달의 시맨틱 특징을 강화하고 CNN으로 심층 융합함으로써, $\text{MIR-Flickr25K}$, $\text{NUS-WIDE}$, $\text{MS COCO}$ 데이터셋에서 기존 $\text{SOTA}$ 모델들을 상회하는 검색 성능($\text{mAP}$)을 달성하였다. 이 연구는 향후 Mamba 네트워크가 멀티미디어 검색 및 이종 데이터 융합 분야에서 매우 강력한 도구가 될 수 있음을 보여준다.