# Zero-shot Classification using Hyperdimensional Computing

Samuele Ruffino, Geethan Karunaratne, Michael Hersche, Luca Benini, Abu Sebastian, and Abbas Rahimi (2024)

## 🧩 Problem to Solve

본 논문은 학습 과정에서 한 번도 본 적 없는 새로운 클래스(unseen classes)의 데이터를 분류해야 하는 Zero-shot Learning (ZSL) 문제, 그 중에서도 Zero-shot Classification (ZSC)을 해결하고자 한다. 특히 세밀한 분류가 필요한 Fine-grained Classification 작업의 경우, 고품질의 레이블링된 데이터를 대량으로 구축하는 데 막대한 비용과 시간이 소요된다는 점이 주요한 병목 현상으로 작용한다.

따라서 본 연구의 목표는 새로운 클래스에 대한 보조 설명서(auxiliary descriptor) 형태의 속성(attribute) 집합을 활용하여, 모델의 파라미터 수를 획기적으로 줄이면서도 높은 분류 정확도를 유지하는 효율적인 ZSC 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Hyperdimensional Computing (HDC)의 원리를 활용하여 속성 인코더(attribute encoder)를 구성하는 것이다. 구체적인 기여 사항은 다음과 같다.

- **HDC-ZSC 아키텍처 제안**: 학습 가능한 이미지 인코더(Image Encoder), HDC 기반의 고정된(stationary) 속성 인코더(Attribute Encoder), 그리고 두 임베딩 간의 유사도를 측정하는 커널(Similarity Kernel)로 구성된 하이브리드 구조를 제안한다.
- **3단계 학습 방법론**: 단순히 ZSC를 학습시키는 것이 아니라, (1) 일반적인 이미지 분류 사전 학습, (2) 도메인 특화 속성 추출(attribute extraction) 학습, (3) 최종 ZSC 분류 학습으로 이어지는 단계적 파이프라인을 통해 정확도를 향상시킨다.
- **파라미터 효율성 및 성능 입증**: CUB-200 데이터셋에서 기존 비생성적(non-generative) 방식보다 적은 파라미터(약 1.72$\sim$1.85배 적음)를 사용하면서도 더 높은 top-1 정확도(63.8%)를 달성하여 Pareto 최적 성능을 보였다.

## 📎 Related Works

ZSL 접근 방식은 크게 비생성적(non-generative) 방식과 생성적(generative) 방식으로 나뉜다.

- **비생성적 방식**: 이미지와 속성 설명 간의 매핑 함수를 탐색하고, 대조 손실(contrastive loss) 등을 통해 두 영역을 정렬하는 방식이다. ES-ZSL과 CLIP이 대표적이며, 본 논문은 이들과 유사한 선형 호환성(linear compatibility) 접근 방식을 따른다.
- **생성적 방식**: 보조 설명서를 기반으로 가상의 unseen class 샘플을 생성하여 문제를 Few-shot Learning 문제로 변환하는 방식이다. 성능은 높을 수 있으나 모델의 크기가 매우 크고 계산 비용이 많이 든다는 한계가 있다.

본 연구는 HDC를 도입하여 속성 인코더를 고정된 이진 코드북(binary codebooks) 형태로 설계함으로써, 기존 비생성적 방식보다 효율적인 메모리 사용과 더 높은 정확도를 동시에 달성하여 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

HDC-ZSC는 입력 이미지 $x$를 임베딩 공간으로 투영하는 이미지 인코더 $\gamma(\cdot)$와, 클래스 속성 $a_i$를 동일한 차원의 임베딩으로 변환하는 속성 인코더 $\phi(\cdot)$로 구성된다. 최종 분류는 두 임베딩의 코사인 유사도(cosine similarity)를 통해 결정된다.

$$ \text{cossim}(\gamma(X), \phi(A)) = \frac{1}{K} \frac{\gamma(X)^T \cdot \phi(A)}{\|\gamma(X)\| \|\phi(A)\|} $$

여기서 $K$는 학습 가능한 온도 스케일링(temperature scaling) 파라미터이다. 예측 클래스 $\hat{y}$는 다음과 같이 결정된다.

$$ \hat{y} = \arg \max_{i \in \{1, \dots, C\}} \text{cossim}(\gamma(x), \phi(a_i)) $$

### 2. HDC 기반 속성 인코더

속성 인코더는 메모리 효율을 위해 모든 속성 조합을 개별적으로 저장하는 대신, 두 가지 코드북을 사용한다.

- **속성 그룹 코드북 (Attribute groups codebook)**: $\{g_1, g_2, \dots, g_G\}$
- **속성 값 코드북 (Attribute values codebook)**: $\{v_1, v_2, \dots, v_V\}$

특정 속성 벡터 $b_x$는 변수 바인딩(variable binding) 연산 $\odot$을 통해 실시간으로 생성된다. 이진 하이퍼벡터의 경우 XOR 연산을, 양극성(bipolar, $\{-1, +1\}$) 하이퍼벡터의 경우 요소별 곱셈(element-wise multiplication)을 수행한다.

$$ b_x = g_y \odot v_z $$

이 방식은 CUB-200 데이터셋 기준 메모리 요구량을 71% 감소시켜, 단 17 KB만으로 속성 벡터를 표현할 수 있게 한다.

### 3. 학습 절차 (3-Phase Training)

- **Phase I (사전 학습)**: ResNet50 백본 네트워크를 ImageNet1K 데이터셋으로 사전 학습하여 일반적인 특징 추출 능력을 확보한다.
- **Phase II (속성 추출 학습)**: 이미지 인코더의 출력과 HDC 기반 속성 딕셔너리 간의 유사도를 측정하고, 가중치 적용 이진 교차 엔트로피 손실(weighted BCEL)을 사용하여 속성 예측 능력을 학습시킨다. 이때 속성 인코더의 코드북은 고정된다.
- **Phase III (ZSC 미세 조정)**: 이미지 인코더의 가중치를 ZSC 작업에 맞춰 미세 조정한다. 클래스 속성 행렬 $A$와 속성 벡터 행렬 $B$의 곱($\phi = A \times B$)으로 클래스 임베딩을 생성하고, 클래스 레이블과의 교차 엔트로피 손실(CEL)을 사용하여 학습한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: CUB-200-2011 (200종의 새 이미지, 312차원 시각 속성 포함).
- **평가 지표**: 속성 추출 작업에서는 Top-1% 정확도와 WMAP(Weighted Mean Average Precision)를 사용하였고, ZSC 작업에서는 Top-1 및 Top-5 정확도를 사용하였다.

### 2. 정량적 결과

- **속성 추출(Attribute Extraction)**: WMAP 기준 Finetag(+4.14%), Top-1% 정확도 기준 A3M(+36.71%)보다 우수한 성능을 보였다.
- **ZSC 성능**: CUB-200 데이터셋에서 **Top-1 정확도 63.8%**를 달성하였다.
- **효율성 비교**: 비생성적 방식인 ES-ZSL 대비 정확도는 9.9% 향상되었으나, 파라미터 수는 1.72배 더 적게 사용하였다. 생성적 방식과 비교했을 때 파라미터 수를 1.75$\sim$2.58배 줄이면서도 경쟁력 있는 정확도를 유지하였다.

## 🧠 Insights & Discussion

본 논문은 HDC의 고차원 벡터 공간 특성을 이용하여 속성 표현을 매우 콤팩트하게 유지하면서도, 다단계 학습 전략을 통해 이미지-속성 간의 정렬(alignment)을 효과적으로 수행할 수 있음을 입증하였다.

특히 주목할 점은 파라미터 수와 정확도 사이의 Trade-off 관계에서 HDC-ZSC가 Pareto 프런트(Pareto front)에 위치한다는 것이다. 이는 모델의 경량화와 성능 향상이 동시에 가능하다는 것을 의미하며, 이는 전력 소모가 제한적인 엣지 디바이스(edge devices)나 비-폰 노이만(non-von Neumann) 가속기 기반의 하드웨어 구현에 매우 유리한 조건이 된다.

다만, 본 논문은 ResNet50 기반의 이미지 인코더를 사용하였으나, 더 가벼운 백본 네트워크를 사용했을 때의 성능 저하 여부나 다른 도메인의 데이터셋에서의 일반화 성능에 대해서는 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 Hyperdimensional Computing (HDC)의 고정된 이진 코드북을 속성 인코더에 도입하여 매우 경량화된 Zero-shot Classification 모델인 **HDC-ZSC**를 제안하였다. 3단계의 전략적 학습 과정을 통해 기존 비생성적 SOTA 모델보다 적은 파라미터로 더 높은 정확도(CUB-200 기준 63.8%)를 달성하였으며, 이는 향후 저전력 임베디드 시스템에서의 ZSC 구현을 위한 핵심적인 기반이 될 가능성이 높다.
