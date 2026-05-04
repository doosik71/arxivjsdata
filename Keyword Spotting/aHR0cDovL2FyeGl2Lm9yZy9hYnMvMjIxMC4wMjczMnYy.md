# FULLY UNSUPERVISED TRAINING OF FEW-SHOT KEYWORD SPOTTING

Dongjune Lee, Minchan Kim, Sung Hwan Mun, Min Hyun Han, Nam Soo Kim (2022)

## 🧩 Problem to Solve

기존의 키워드 검출(Keyword Spotting, KWS) 시스템은 미리 정의된 특정 키워드들에 대해 최적화되어 있어, 새로운 키워드를 추가하려면 대량의 레이블링된 데이터셋이 필요하며 시스템의 유연성이 떨어진다는 문제가 있다. 이를 해결하기 위해 적은 수의 샘플만으로 새로운 키워드를 등록할 수 있는 Few-Shot KWS(FS-KWS) 연구가 진행되어 왔으나, 여전히 모델의 일반화 성능을 확보하기 위해서는 수많은 다양한 키워드가 포함된 대규모 레이블링 데이터셋이 필수적이다.

본 논문의 목표는 이러한 비용 효율적이지 못한 데이터 수집 및 레이블링 과정을 완전히 제거하고, 오직 합성 데이터(Synthetic Data)만을 사용하여 학습된 완전히 비지도 방식(Fully Unsupervised)의 FS-KWS 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 레이블이 없는 음성 코퍼스에서 추출한 의사 음소(Pseudo Phoneme)를 이용해 음성을 합성하고, 이를 Metric Learning의 학습 데이터로 활용하는 것이다. Metric Learning은 본질적으로 텍스트 레이블보다 '동일한 의미/발음의 샘플들이 임베딩 공간에서 가깝게 위치해야 한다'는 구조적 특성을 가지므로, 텍스트 레이블 없이도 동일한 발음을 가진 다양한 화자의 음성 샘플(Multi-view samples)만 확보된다면 학습이 가능하다는 직관에 기반한다.

## 📎 Related Works

기존의 KWS 모델들은 주로 분류(Classification) 문제로 접근하였으나, 이는 미등록 단어(Unknown words) 처리에 취약하고 새로운 키워드 추가가 어렵다는 한계가 있다. 이를 극복하기 위해 Triplet Loss나 Prototypical Loss와 같은 Metric Learning 기반의 접근 방식이 제안되었으며, 이는 거리 측정(Distance metrics)을 통해 타겟 키워드를 검출함으로써 유연성을 확보하였다.

또한, 데이터 부족 문제를 해결하기 위해 TTS(Text-to-Speech)나 GAN을 이용한 데이터 증강(Data Augmentation) 연구들이 진행되었다. 하지만 기존 연구들은 주로 실제 데이터셋을 보조하는 수단으로 합성 데이터를 사용한 반면, 본 논문은 실제 레이블링된 데이터셋을 완전히 배제하고 합성 데이터만으로 모델을 학습시킨다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 프레임워크는 $\text{wav2vec2.0}$의 표현력을 이용해 추출한 의사 음소(Pseudo Phoneme)를 기반으로 한 $\text{pseudo-TTS}$ 모델을 데이터 생성기로 사용하며, 이를 통해 생성된 데이터를 Prototypical Networks로 학습시킨다.

### 2. 학습 과정 및 목표 (Training)

**가. 학습 목표 (Training Objective)**
본 연구는 $N$-way $K$-shot 분류 시나리오를 채택한 Prototypical Networks를 사용한다. 각 클래스 $n$에 대해 $K$개의 서포트 샘플과 1개의 쿼리 샘플이 주어진다.

- **프로토타입 계산**: 클래스 $n$의 프로토타입 $c_n$은 인코더 $f_\phi$를 통해 투영된 서포트 샘플들의 평균으로 계산한다.
$$c_n = \frac{1}{K} \sum_{k=1}^K f_\phi(x^{n,k})$$

- **확률 계산**: 쿼리 샘플 $x$가 클래스 $n$에 속할 확률은 프로토타입과의 유클리드 거리($\text{dist}$)를 기반으로 Softmax 함수를 통해 계산한다.
$$p(y=n|x) = \frac{\exp(-\text{dist}(f_\phi(x), c_n))}{\sum_{n'=1}^N \exp(-\text{dist}(f_\phi(x), c_{n'}))}$$

- **손실 함수**: 최종 학습은 쿼리 샘플에 대한 교차 엔트로피 손실(Cross-Entropy Loss)을 최소화하는 방향으로 진행된다.
$$L = -\frac{1}{N} \sum_{n=1}^N \log p(y=n|x^{n,K+1})$$

**나. pseudo-TTS를 이용한 데이터 생성**
텍스트 레이블 없이 데이터를 생성하기 위해 다음과 같은 절차를 거친다.

1. 레이블이 없는 음성 코퍼스에서 $\text{wav2vec2.0}$ 임베딩을 추출하고, 이를 $k$-means 클러스터링하여 의사 음소(Pseudo Phoneme) 시퀀스를 얻는다.
2. 이 시퀀스를 무작위로 잘라(Random crop) 임의의 길이($L_{\min}, L_{\max}$)를 가진 의사 키워드를 생성한다.
3. $\text{pseudo-TTS}$ 모델에 의사 음소 시퀀스와 참조 음성(Reference speech)을 입력하여, 동일한 발음을 가지되 화자와 운율이 다른 다양한 합성 음성을 생성한다.
4. 합성 음성과 실제 음성 간의 도메인 차이를 줄이기 위해 볼륨 조절, 잔향(Reverberation) 추가, 노이즈 주입 등의 데이터 증강(Augmentation)을 적용한다.

**다. 효율적 학습을 위한 데이터 버퍼**
매 반복(Iteration)마다 수천 초의 음성을 합성하는 것은 계산 비용이 너무 크므로, 데이터 버퍼(Data Buffer)를 사용한다. 버퍼에 $M_{\text{buffer}}$개의 클래스를 미리 생성해 두고, 매 반복마다 일부 클래스만 업데이트하는 방식으로 학습 속도를 높였다.

### 3. 추론 과정 (Inference)

추론 시에는 등록된 타겟 키워드 샘플들로 프로토타입 $c_n$을 계산한다. 입력 쿼리 $x$에 대해 가장 가까운 프로토타입을 찾고, 그 거리가 특정 임계값 $D_{\text{th}}$보다 작으면 해당 키워드로 판정하고, 그렇지 않으면 '알 수 없음(Unknown)' 클래스로 분류한다.
$$\hat{y} = \arg\min_n \text{dist}(f_\phi(x), c_n)$$
$$y_{\text{pred}} = \begin{cases} \hat{y}, & \text{if } \text{dist}(f_\phi(x), c_{\hat{y}}) < D_{\text{th}} \\ N+1, & \text{else} \end{cases}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Google Speech Commands v1 (GSCv1), Multilingual Spoken Word Corpus (MSWC-en).
- **모델 아키텍처**: $\text{TC-Resnet}$을 기반으로 하되, Average Pooling 레이어를 $\text{GRU}$ 레이어로 교체하여 시간적 동적 특성을 더 잘 포착하도록 수정하였다. 입력은 40차원 MFCC, 출력 임베딩은 192차원이다.
- **비교 대상**: $\text{Supervised (MSWC)}$, $\text{Unsupervised (w/o aug)}$, $\text{Unsupervised (w aug)}$ (제안 방법).
- **지표**: $\text{Acc (target)}$, $\text{Acc (total)}$, $\text{AUROC}$.

### 2. 주요 결과

- **데이터 증강의 중요성**: $\text{Unsupervised (w aug)}$ 모델이 $\text{w/o aug}$ 모델보다 모든 지표에서 압도적으로 높은 성능을 보였다. 이는 단순한 깨끗한 합성 음성보다 적절한 증강이 가해진 데이터가 실제 음성 도메인과의 간극을 줄이는 데 핵심적임을 시사한다.
- **지도 학습과의 비교**:
  - **GSCv1**: 제안 방법이 지도 학습 모델과 비슷하거나 오히려 약간 더 높은 성능을 기록하였다. 이는 레이블링된 데이터 없이도 충분한 일반화 성능을 얻을 수 있음을 입증한다.
  - **MSWC**: 지도 학습 모델이 더 높은 성능을 보였으나, 이는 학습 데이터와 테스트 데이터 간의 도메인 불일치로 인한 결과로 분석된다.
- **정성적 분석**: t-SNE 시각화 결과, 동일 클래스의 임베딩들이 서로 가깝게 클러스터링되는 것을 확인하였으며, 이는 제안한 비지도 학습 방식이 유의미한 특징 추출을 수행하고 있음을 보여준다.

## 🧠 Insights & Discussion

본 논문은 고충실도(High-fidelity) 및 제어 가능한 최신 음성 합성 시스템이 FS-KWS 분야에서 레이블링된 데이터셋을 대체할 가능성이 있음을 실험적으로 증명하였다. 특히 Metric Learning의 특성을 이용해 '텍스트'라는 명시적 레이블 없이 '발음(Phonetics)' 정보만으로 학습이 가능하다는 점이 매우 인상적이다.

다만, 합성 음성과 실제 음성 사이의 도메인 갭(Domain Gap)은 여전히 해결해야 할 과제이다. 본 논문에서는 단순한 증강 기법으로 이를 완화했으나, 향후에는 더 정교한 도메인 적응(Domain Adaptation) 알고리즘이나 합성 음성의 다양성(화자, 속도, 운율의 정밀 제어)을 높이는 연구가 병행되어야 할 것이다. 또한, 아키텍처 설계의 개선이나 다른 Metric Learning 기법의 적용을 통해 성능을 더 끌어올릴 여지가 있다.

## 📌 TL;DR

본 연구는 $\text{pseudo-TTS}$와 $\text{Prototypical Networks}$를 결합하여, 실제 레이블링된 데이터 없이 오직 합성 데이터만으로 학습하는 **완전 비지도 방식의 Few-Shot KWS** 프레임워크를 제안한다. 실험 결과, 적절한 데이터 증강이 동반될 경우 실제 데이터로 학습한 지도 학습 모델에 근접하는 성능을 보였으며, 이는 향후 사용자 정의 키워드 검출 시스템 구축 시 데이터 수집 비용을 획기적으로 줄일 수 있는 가능성을 제시한다.
