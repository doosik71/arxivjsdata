# Mamba-based Segmentation Model for Speaker Diarization

Alexis Plaquet, Naohiro Tawara, Marc Delcroix, Shota Horiguchi, Atsushi Ando, and Shoko Araki (2024)

## 🧩 Problem to Solve

본 논문은 오디오 녹음 파일에서 "누가 언제 말했는가"를 식별하는 화자 분리(Speaker Diarization) 문제, 특히 End-to-End Neural Diarization (EEND) 기반의 세그멘테이션 모델의 한계를 해결하고자 한다.

기존의 EEND 모델들은 주로 두 가지 아키텍처에 의존해 왔다. 첫째는 BiLSTM과 같은 순환 신경망(RNN)으로, 이는 순차적 데이터 처리에는 적합하지만 메모리 제한으로 인해 장기 의존성(long-term dependencies)을 캡처하는 능력이 부족하다. 둘째는 Attention 기반 모델로, 장기 의존성 모델링 능력은 뛰어나지만 시퀀스 길이에 따라 계산 복잡도가 제곱으로 증가하는 $O(L^2)$의 비용 문제가 발생하며, 많은 경우 데이터의 순차적 특성을 반영하는 positional embedding을 적절히 활용하지 못하는 한계가 있다.

따라서 본 연구의 목표는 RNN의 순차적 처리 능력과 Attention의 강력한 메모리 성능을 동시에 갖추면서도 선형 복잡도를 가진 Mamba 아키텍처를 화자 분리 시스템에 도입하여, 긴 오디오 윈도우를 효율적으로 처리하고 전체적인 분리 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba(State Space Model, SSM)를 화자 분리를 위한 세그멘테이션 모델에 적용하여 기존의 BiLSTM 및 Attention 기반 모델보다 우수한 성능을 입증한 것이다.

가장 중심적인 아이디어는 Mamba의 효율적인 순차 처리 능력과 선택적 기억(selective memorization) 능력을 활용해, 기존 모델들이 감당하기 어려웠던 더 긴 로컬 윈도우(local window)를 사용할 수 있게 하는 것이다. 윈도우 크기가 커지면 세그멘테이션 자체의 난이도는 상승하지만, 화자 임베딩(speaker embedding) 추출을 위한 샘플이 더 많이 확보되어 결과적으로 클러스터링 단계의 신뢰도가 높아지고 전체 시스템의 화자 분리 정확도가 향상된다는 직관을 제시하였다.

## 📎 Related Works

화자 분리 분야에서는 전통적인 Vector Clustering(VC) 방식과 직접적으로 화자 활성 구간을 예측하는 EEND 방식이 사용되어 왔다. 최근에는 이 두 방식의 장점을 결합하여, 짧은 윈도우 단위로 EEND를 적용한 후 클러스터링으로 화자 ID를 정렬하는 하이브리드 방식인 EEND-VC(예: `pyannote.audio` 파이프라인)가 주류를 이루고 있다.

최근 Mamba와 같은 State Space Model(SSM)은 음성 분리(speech separation), 음성 향상(speech enhancement) 및 오디오 표현 학습(audio representation learning) 분야에서 Transformer와 경쟁 가능한 성능을 보이면서도 더 적은 파라미터와 빠른 속도를 보여주었다. 그러나 본 논문은 Mamba를 화자 분리 작업에 적용한 첫 번째 연구임을 명시하며, 기존의 BiLSTM 및 Attention 기반 세그멘테이션 모델과의 차별성을 강조한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (EEND-VC Pipeline)

본 연구는 다음과 같은 단계로 구성된 파이프라인을 사용한다.

1. **특징 추출(Feature Extraction)**: 사전 학습된 WavLM Base 모델을 사용하여 오디오에서 768차원의 특징을 추출한다. 이 모델은 동결(frozen) 상태로 사용된다.
2. **로컬 EEND 세그멘테이션**: 추출된 특징을 Mamba 또는 BiLSTM 기반의 프로세싱 모듈에 통과시켜 각 프레임별 화자 활성화 여부를 예측한다.
3. **임베딩 추출 및 클러스터링**: 세그멘테이션 결과에서 겹치지 않는 구간을 선택해 ResNet 기반 모델로 화자 임베딩을 추출하고, Agglomerative Hierarchical Clustering을 통해 최종 화자 ID를 할당한다.

### 2. Mamba 기반 세그멘테이션 모델 아키텍처

제안된 Mamba 기반 모듈의 상세 구조는 다음과 같다.

- **입력 층**: WavLM의 768차원 특징을 Linear 레이어를 통해 256차원으로 축소하여 모델의 파라미터 수를 최적화한다.
- **프로세싱 모듈**: 7개의 $\text{ExternalBidirectional Mamba}$ 블록을 체인 형태로 연결하여 구성한다. 이때 내부 상태 차원(state dimension) $d_{state}$는 64로 설정하였다.
- **출력 층**: 두 개의 Linear 레이어(hidden size 128)를 거쳐 최종 출력 차원 $C$로 변환한다.

### 3. 학습 목표 및 손실 함수

모델은 두 가지 출력 표현 방식을 비교 분석하였다.

- **Multilabel**: 각 레이블이 개별 화자의 활성화 여부를 나타내며, Permutation-free Binary Cross Entropy (BCE) 손실 함수를 사용한다.
- **Multiclass Powerset**: 가능한 모든 화자 조합을 하나의 클래스로 인코딩하며, Permutation-free Cross Entropy 손실 함수를 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: NOTSOFAR-1, MSDWild, VoxConverse 등 8개의 데이터셋을 합친 compound dataset으로 학습하였으며, DIHARD III 데이터셋을 통해 도메인 외(out-of-domain) 평가를 수행하였다.
- **비교 대상**: BiLSTM 기반 모델, Attention 기반 모델.
- **주요 지표**: 화자 분리 오류율(Diarization Error Rate, DER).

### 2. 주요 결과 분석

- **윈도우 크기의 영향**: Oracle Clustering(이상적인 클러스터링) 환경에서는 윈도우 크기가 커질수록 DER이 증가한다. 하지만 전체 파이프라인 관점에서는 Mamba 모델이 $W=30\text{s}$에서 최적의 성능을 보였다. 이는 윈도우가 길어질수록 세그멘테이션 오류는 늘어나지만, 임베딩 추출의 품질이 향상되어 전체 DER이 낮아지는 트레이드-오프 관계가 존재함을 시사한다.
- **아키텍처 비교**: 동일한 윈도우 크기($W=10\text{s}$)에서 Mamba는 BiLSTM보다 일관되게 우수한 성능을 보였다. 특히 Attention 기반 모델은 학습이 매우 어렵고 비용이 많이 들었으나, Mamba는 훨씬 적은 학습 데이터로도 더 높은 성능을 달성하였다.
- **정량적 성과**: 제안된 Mamba 기반 시스템은 RAMC, AISHELL-4, MSDWILD 데이터셋에서 SOTA(State-of-the-art) 성능을 달성하였으며, DIHARD III와 AMI에서도 경쟁력 있는 결과를 보여주었다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

본 연구는 Mamba가 RNN의 순차적 특성과 Attention의 장기 기억 능력을 결합함으로써, 화자 분리 시스템에서 가장 병목이 되는 '윈도우 크기 결정' 문제에 새로운 해결책을 제시하였다. 특히 BiLSTM이 감당하지 못하는 긴 시퀀스에서도 효율적으로 작동하여 더 신뢰할 수 있는 임베딩 추출을 가능하게 했다는 점이 핵심적인 강점이다.

### 2. 한계 및 비판적 해석

- **손실 함수의 역설**: Oracle clustering 결과에서는 Powerset loss가 유리하게 나타났으나, 최종 파이프라인 DER에서는 Multilabel loss가 더 좋은 성능을 보였다. 분석 결과, Powerset loss가 False Alarm은 줄이지만 화자 혼동(speaker confusion)을 증가시켜 결과적으로 임베딩 품질을 떨어뜨린다는 점이 발견되었다. 이는 로컬 지표의 개선이 반드시 전체 시스템의 성능 향상으로 이어지지 않음을 보여준다.
- **파라미터 수의 영향**: Mamba 모듈이 BiLSTM보다 파라미터 수가 많음에도 불구하고, 파라미터 수를 동일하게 맞춘 실험에서도 Mamba가 여전히 우위에 있었다. 이는 단순한 모델 크기의 차이가 아니라 아키텍처 자체의 효율성에서 기인한 결과라고 볼 수 있다.

## 📌 TL;DR

본 논문은 화자 분리(Speaker Diarization)의 세그멘테이션 단계에 **Mamba(SSM)** 아키텍처를 도입하여, 기존 BiLSTM의 메모리 한계와 Attention의 계산 복잡도 문제를 동시에 해결하였다. Mamba를 통해 더 긴 처리 윈도우를 사용할 수 있게 됨으로써 화자 임베딩의 정확도를 높였고, 결과적으로 3개의 주요 데이터셋에서 **SOTA 성능**을 기록하였다. 이 연구는 향후 긴 오디오 시퀀스를 처리해야 하는 다양한 음성 처리 작업에서 Mamba가 강력한 대안이 될 수 있음을 시사한다.
