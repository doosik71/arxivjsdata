# Simple and Effective Unsupervised Speech Translation

Changhan Wang, Hirofumi Inaguma, Peng-Jen Chen, Ilia Kulikov, Yun Tang, Wei-Ning Hsu, Michael Auli, Juan Pino (2022)

## 🧩 Problem to Solve

본 논문은 음성 번역(Speech Translation, ST) 모델 학습에 필요한 레이블링된 데이터(labeled data)의 극심한 부족 문제를 해결하고자 한다. 일반적인 음성 인식(ASR)이나 기계 번역(MT) 작업보다 음성 번역은 두 가지 서로 다른 언어의 정렬된 데이터가 동시에 필요하기 때문에 데이터 희소성 문제가 더욱 심각하다. 특히 전 세계 7,000개 이상의 언어 중 극소수만을 제외한 대부분의 언어는 이러한 병렬 말뭉치가 거의 존재하지 않는 저자원(low-resource) 환경에 처해 있다.

따라서 본 연구의 목표는 레이블링된 데이터 없이, 즉 비지도 학습(unsupervised learning) 방식만을 활용하여 효과적인 음성-텍스트 번역(S2TT) 및 음성-음성 번역(S2ST) 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최신 비지도 학습 기술들(비지도 음성 인식, 비지도 기계 번역, 비지도 음성 합성)을 직렬로 연결(cascade)하여 합성 데이터(synthetic data)를 생성하고, 이를 통해 가짜 레이블(pseudo-labels)을 만들어 최종적으로 종단간(end-to-end) 모델을 학습시키는 것이다.

주요 기여 사항은 다음과 같다.

1. **비지도 캐스케이드 의사 레이블링(Unsupervised Cascaded Pseudo-Labeling):** ASR $\rightarrow$ TDN $\rightarrow$ MT $\rightarrow$ TTS 순서의 파이프라인을 통해 학습 데이터를 생성하고, 이를 이용해 종단간 S2TT 및 S2ST 모델을 학습시키는 프레임워크를 제안한다.
2. **비지도 도메인 적응(Unsupervised Domain Adaptation):** 사전 학습된 음성 모델(wav2vec 2.0)이 대상 도메인이나 언어와 일치하지 않을 때 발생하는 성능 저하를 막기 위해, 레이블 없는 도메인 내 데이터만을 이용해 모델을 적응시키는 기법을 제시한다.
3. **텍스트 역정규화(Text De-normalization, TDN) 도입:** ASR의 출력(정규화된 텍스트)과 MT의 입력(비정규화된 텍스트) 사이의 괴리를 메우기 위해 TDN 단계를 추가하여 번역 품질을 향상시켰다.

## 📎 Related Works

본 연구는 다음과 같은 기존 비지도 학습 연구들에 기반하고 있다.

- **비지도 음성 인식(Unsupervised ASR):** wav2vec-U 및 wav2vec-U 2.0과 같이 자기지도 학습(self-supervised learning) 표현력을 활용하여 레이블 없이 음성을 인식하는 연구들이 진행되었다.
- **비지도 기계 번역(Unsupervised MT):** mBART와 같이 다국어 텍스트 코퍼스를 활용하여 병렬 말뭉치 없이 언어 간 정렬을 학습하는 방식이 제안되었다.
- **비지도 음성 합성(Unsupervised TTS):** 비지도 ASR 모델로 생성한 레이블을 사용하여 TTS를 학습시키는 방식이 연구되었다.

기존의 종단간 음성 번역 모델들은 대규모의 병렬 데이터가 필수적이었으나, 본 논문은 이러한 의존성을 완전히 제거하고 비지도 모델들의 조합을 통해 데이터를 생성하여 학습한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

전체 시스템은 크게 세 단계의 파이프라인으로 구성된다.

### 1. 비지도 도메인 적응 (Unsupervised Adaptation of wav2vec 2.0)

사전 학습된 wav2vec 2.0 모델이 타겟 언어/도메인과 일치하지 않을 경우 학습 수렴이 어렵다. 이를 해결하기 위해 다음 과정을 거친다.

- 레이블 없는 도메인 내 음성 데이터의 표현(representation)을 추출하여 $k$-means 클러스터링을 수행하고, 각 프레임에 클러스터 ID를 할당한다.
- 연속된 동일 ID를 병합하여 이산적인 레이블(discrete labels)을 생성한다.
- 이 레이블을 사용하여 CTC(Connectionist Temporal Classification) 목적 함수로 wav2vec 2.0 모델 전체를 미세 조정(fine-tuning)한다.

### 2. 비지도 캐스케이드 의사 레이블링 (Unsupervised Cascaded Pseudo-Labeling)

다음의 모델들을 순차적으로 연결하여 S2TT 및 S2ST를 위한 학습 데이터를 생성한다.

- **Unsupervised ASR:** wav2vec-U 2.0을 사용한다. 학습 안정성을 위해 입력 특성에 가우시안 노이즈를 추가하고, R-Drop 정규화를 적용한다. R-Drop 손실 함수는 서로 다른 드롭아웃 마스크를 가진 두 생성기 $G_1, G_2$ 사이의 KL 발산을 최소화하는 것이다.
  $$L_{rdp} = \frac{1}{2} D_{KL}(G_1(X') || G_2(X')) + \frac{1}{2} D_{KL}(G_2(X') || G_1(X'))$$
- **Unsupervised TDN:** ASR 출력(소문자, 문장부호 없음)을 MT 입력(대소문자 및 문장부호 포함) 형태로 변환한다. mBART 모델을 사용하여 정규화된 텍스트를 원문 형태로 복원하도록 학습한다.
- **Unsupervised MT:** mBART-OBT를 사용하여 소스 언어 텍스트를 타겟 언어 텍스트로 번역한다.
- **Unsupervised TTS (S2ST의 경우):** autoregressive Transformer TTS를 사용하며, 문장 종료(EOS) 예측 성능을 높이기 위해 다음과 같은 R-Drop 스타일의 일관성 손실($L_c$)을 추가한다.
  $$L_c = ||P_{EOS_1}(X) - P_{EOS_2}(X)||_1$$

### 3. 종단간 모델 학습 (End-to-end Model Training)

위 단계에서 생성된 의사 레이블(pseudo-labels)을 사용하여 지도 학습 방식으로 종단간 모델을 학습시킨다.

- **S2TT:** w2v2-mBART 구조를 사용하며, 인코더는 w2vu2-CTC로, 디코더는 mBART-OBT로 사전 학습된다.
- **S2ST:** Spec-T2(Translatotron 2의 변형) 구조를 사용하며, 마찬가지로 인코더와 디코더를 각각 비지도 ASR과 MT 모델로 사전 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 방향:** X-En (Fr, Es, Ru, Et, Lv $\rightarrow$ En) 및 En-X (En $\rightarrow$ Es, Ru, Fr) 방향을 평가하였다.
- **벤치마크:** CoVoST 2, CVSS-C, MuST-C, Libri-Trans를 사용하였다.
- **지표:** BLEU (번역 품질), WER (음성 인식/합성 오류율), PER (음소 오류율)을 측정하였다.

### 주요 결과

- **S2TT 성능:** CoVoST 2 벤치마크에서 비지도 종단간 모델이 2년 전의 최신 지도 학습 모델(pre-training 없는 모델)보다 평균 5.0 BLEU 높은 성능을 보였다. 특히 Libri-Trans(En-Fr)에서는 이전 비지도 SOTA 대비 3.2 BLEU 향상된 결과를 얻었다.
- **S2ST 성능:** CVSS-C 벤치마크에서 비지도 모델이 지도 학습 베이스라인보다 평균 0.8 BLEU 낮은 수준까지 근접하며 경쟁력 있는 성능을 보였다.
- **도메인 적응 효과:** wav2vec 2.0의 비지도 적응 기법을 적용했을 때, 특히 저자원 언어(Et, Lv)에서 모델의 수렴 실패 문제가 해결되고 성능이 크게 향상되었다.
- **TDN의 효과:** 텍스트 역정규화(TDN)를 적용했을 때 BLEU 점수가 평균 4.1 상승하였으며, 이는 단순히 문장부호를 복원하는 것을 넘어 전반적인 번역 내용의 질을 향상시킴을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 비지도 학습 모델들을 적절히 조합하여 고품질의 합성 데이터를 생성하고, 이를 통해 종단간 모델을 학습시키는 전략이 유효함을 입증하였다. 특히 다음과 같은 통찰을 제공한다.

첫째, **저자원 환경에서의 모델 선택 문제**이다. 데이터가 극도로 적은 언어(Ru, Et, Lv)의 경우 종단간 모델은 합성 데이터 양이 적어 오버피팅(overfitting)이 발생하기 쉽다. 반면 캐스케이드 모델은 텍스트 기반의 비지도 MT가 더 많은 양의 단일 언어 텍스트 데이터를 활용할 수 있기 때문에 저자원 설정에서 더 강건한 성능을 보인다.

둘째, **에러 누적(Error Propagation)의 문제**이다. S2ST의 경우 S2TT보다 성능 향상 폭이 적은데, 이는 ASR $\rightarrow$ MT $\rightarrow$ TTS로 이어지는 과정에서 각 단계의 오류가 누적되어 최종 합성 데이터의 품질이 저하되기 때문으로 분석된다.

셋째, **사전 학습 모델의 도메인 적응 중요성**이다. 저자원 언어일수록 사전 학습된 모델의 도메인 불일치가 학습 성패를 결정짓는 핵심 요인이며, 단순한 $k$-means 클러스터링 기반의 적응만으로도 이를 효과적으로 해결할 수 있음을 보여주었다.

## 📌 TL;DR

본 논문은 레이블링된 데이터 없이 **비지도 ASR $\rightarrow$ TDN $\rightarrow$ MT $\rightarrow$ TTS**를 연결한 파이프라인으로 의사 레이블을 생성하고, 이를 통해 종단간 음성 번역 모델을 학습시키는 방법론을 제안한다. 특히 wav2vec 2.0의 비지도 도메인 적응 기법을 통해 저자원 언어에서도 학습 가능성을 높였으며, 기존 비지도 SOTA 및 과거의 지도 학습 모델들을 능가하는 성능을 달성하였다. 이 연구는 병렬 데이터 구축이 어려운 수많은 희귀 언어의 음성 번역 시스템 구축에 중요한 기틀을 제공할 것으로 기대된다.
