# kNN-CTC: ENHANCING ASR VIA RETRIEVAL OF CTC PSEUDO LABELS

Jiaming Zhou, Shiwan Zhao, Yaqi Liu, Wenjia Zeng, Yong Chen, Yong Qin (2024)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템에서 Retrieval-Augmented(검색 증강) 모델을 적용할 때 발생하는 데이터스토어(Datastore) 구축의 어려움을 해결하고자 한다. 자연어 처리(NLP) 분야에서는 $k$-nearest neighbors (kNN) 모델을 통해 언어 모델의 성능을 크게 향상시킨 사례가 많지만, ASR에서는 다음과 같은 두 가지 핵심적인 문제로 인해 세밀한(fine-grained) 데이터스토어 구축이 어려웠다.

첫째, 오디오 프레임과 텍스트 문자 간의 정밀한 정렬(alignment) 정보가 부족하여, 각 프레임에 대응하는 정답 라벨(ground-truth labels)을 확보하기 어렵다는 점이다. 둘째, 오디오를 프레임 단위로 처리할 경우 생성되는 데이터의 양이 방대하여 데이터스토어의 크기가 지나치게 커진다는 점이다. 따라서 본 연구의 목표는 정밀한 정렬 없이도 프레임 수준의 오디오-텍스트 쌍을 구축하고, 효율적으로 데이터스토어 크기를 줄이면서 ASR 성능을 높이는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Connectionist Temporal Classification (CTC)의 pseudo labels를 활용하여 정밀한 정렬 과정 없이도 프레임 수준의 키-값(key-value) 쌍을 생성하는 것이다. 또한, CTC의 특성상 빈번하게 발생하는 blank 프레임을 전략적으로 제거하는 'skip-blank' 전략을 도입하여 데이터스토어의 용량을 획기적으로 줄이면서도 효율적인 검색을 가능하게 하였다. 이를 통해 사전 학습된 CTC 기반 ASR 시스템에 kNN 검색 메커니즘을 결합하여 전반적인 인식 성능을 향상시켰다.

## 📎 Related Works

기존의 kNN 기반 ASR 연구들은 GMM-HMM 또는 DNN-HMM 방식에 kNN을 결합하여 성능을 개선하려 했다. 최근에는 Transducer 기반 ASR 모델에서 외부 텍스트 코퍼스를 검색하여 부분적인 가설을 완성하는 방식이 제안되었으나, 이는 텍스트 모달리티만을 강화하는 kNN 언어 모델의 범주에 머물러 있었다. 일부 연구에서는 TTS(Text To Speech)를 이용해 오디오를 생성하고 이를 데이터스토어로 활용하여 Contextual ASR을 구현하기도 했지만, 이는 구절(phrase) 단위의 coarse-grained 접근 방식이었다는 한계가 있다. 반면, 본 연구는 프레임 수준의 fine-grained 데이터스토어를 구축함으로써 기존의 구절 단위나 텍스트 중심의 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. kNN 모델 및 데이터스토어 구축

본 시스템은 사전 학습된 CTC 기반 ASR 모델을 바탕으로 데이터스토어 구축과 후보군 검색(candidate retrieval)의 두 단계로 구성된다.

**데이터스토어 구축 (Datastore Construction):**
정밀한 정렬 정보 대신 CTC pseudo labels를 사용한다. 학습 데이터셋 $S$에 대해, 모델의 마지막 인코더 층의 Feed-Forward Network (FFN) 입력 부분을 키(key)로 선택한다. 각 프레임 $X_i$에 대한 pseudo label $\hat{Y}_i$는 다음과 같이 결정된다.

$$\hat{Y}_i = \arg \max_{Y_i} P^{CTC}(Y_i | X_i)$$

이렇게 얻은 중간 표현 $f(X_i)$를 키 $k_i$로, pseudo label $\hat{Y}_i$를 값 $v_i$로 설정하여 데이터스토어 $(K, V)$를 구축한다.

$$(K, V) = \{(f(X_i), \hat{Y}_i) | X_i \in S\}$$

**후보군 검색 및 융합 (Candidate Retrieval & Fusion):**
추론 단계에서 현재 프레임의 중간 표현 $f(x)$를 쿼리로 사용하여 데이터스토어에서 $k$개의 최근접 이웃 $N$을 검색한다. 검색된 이웃들에 대해 softmax 확률 분포를 계산하며, 식은 다음과 같다.

$$p^{kNN}(y|x) \propto \sum_{(k_i, v_i) \in N, v_i=y} \exp(-d(k_i, f(x))/\tau)$$

여기서 $\tau$는 temperature, $d(\cdot, \cdot)$는 $L_2$ 거리를 의미한다. 최종 출력 분포 $p(y|x)$는 CTC 모델의 출력과 kNN 모델의 출력을 선형 보간하여 산출한다.

$$p(y|x) = \lambda p^{kNN}(y|x) + (1-\lambda) p^{CTC}(y|x)$$

$\lambda$는 두 분포의 기여도를 조절하는 하이퍼파라미터이다.

### 2. Skip-blank 전략

CTC 모델은 특성상 많은 프레임이 `<blank>` 심볼로 할당되는 경향이 있다. 이를 그대로 저장하면 데이터스토어가 불필요하게 커지므로, 구축 단계에서 pseudo label이 `<blank>`인 프레임을 제외하는 'skip-blank' 전략을 사용한다. 추론 시에도 CTC 디코딩 출력을 참고하여 blank 프레임에 대해서는 kNN 검색을 건너뛰어 효율성을 높인다.

### 3. 비지도 도메인 적응 (Cross-domain Adaptation)

레이블이 없는 타겟 도메인 데이터 $X^T$가 주어졌을 때, 소스 도메인에서 학습된 모델을 이용해 $X^T$에 대한 CTC pseudo labels를 생성하고 이를 통해 데이터스토어를 구축한다. 이 과정은 추가적인 모델 학습 없이 데이터스토어 구성만으로 빠르게 도메인 적응을 수행할 수 있게 한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** AISHELL-1, AISHELL-2 (중국어), Libri-Adapt (영어), WenetSpeech (중국어), KeSpeech 방언 데이터셋 등을 사용하였다.
- **기준 모델:** Joint CTC-Attention Conformer 모델을 베이스라인으로 사용하였다.
- **지표:** 영어는 WER(Word Error Rate), 중국어는 CER(Character Error Rate)을 측정하였다.
- **구현 세부사항:** FAISS 라이브러리를 이용해 $k=1024$인 근사 최근접 이웃 검색을 수행하였다.

### 2. 주요 결과

- **In-domain ASR:** 모든 데이터셋에서 kNN-CTC가 vanilla CTC보다 우수한 성능을 보였다. 특히 skip-blank를 적용한 `kNN-CTC (pruned)` 버전은 full 버전과 거의 대등한 성능을 유지하면서도 데이터스토어 크기를 약 84.56% 감소시켰다.
- **Cross-domain ASR:** WenetSpeech를 소스 도메인으로 하여 다양한 중국어 방언 및 일반 중국어 데이터셋에 적용한 결과, kNN-CTC (full)가 CTC 대비 평균 4.06%의 CER 감소를 보였다. Pruned 버전 역시 성능 향상을 보였으나, 도메인 간 차이가 큰 경우 full 버전에 비해 성능이 약간 낮게 나타났다.

## 🧠 Insights & Discussion

**키 위치의 영향:**
연구진은 Conformer 인코더의 세 가지 위치(Encoder output, FFN input after layer norm, FFN input before layer norm)를 실험하였으며, Layer Normalization 이후의 FFN 입력 단계가 가장 최적의 성능을 보임을 확인하였다.

**하이퍼파라미터 $\lambda$의 민감도:**
In-domain 설정에서는 $\lambda > 0$일 때 일관되게 성능이 향상되었으나, Cross-domain 설정에서는 $\lambda$가 0.4보다 클 때 오히려 성능이 하락하였다. 이는 도메인 시프트로 인해 타겟 데이터의 pseudo label 품질이 낮아졌기 때문이며, kNN-CTC의 성능이 데이터스토어에 사용된 pseudo label의 품질에 의존함을 시사한다.

**에러 분석:**
CTC 모델의 주요 에러는 치환(Substitution, S) 에러이다. kNN 메커니즘은 잘못된 non-blank 심볼을 올바른 심볼로 교체함으로써 S 에러를 효과적으로 수정한다. 다만, pruned 버전은 blank 프레임을 제거하므로, blank를 non-blank로 교체해야 하는 삭제(Deletion, D) 에러 수정 능력은 제한적이며, 이로 인해 Cross-domain 상황에서 full 버전보다 약간 낮은 성능을 보인 것으로 분석된다.

## 📌 TL;DR

본 논문은 CTC pseudo labels를 활용해 정밀한 정렬 없이 프레임 수준의 오디오-텍스트 데이터스토어를 구축하고, 이를 kNN 검색과 결합하여 ASR 성능을 높이는 **kNN-CTC**를 제안한다. 특히 **skip-blank** 전략을 통해 데이터스토어의 용량을 80% 이상 획기적으로 줄이면서도 효율적인 추론이 가능함을 입증하였다. 이 방법론은 추가 학습 없이 데이터스토어 교체만으로 빠르게 도메인 적응을 수행할 수 있어, 실용적인 ASR 시스템 확장 및 최적화에 기여할 가능성이 높다.
