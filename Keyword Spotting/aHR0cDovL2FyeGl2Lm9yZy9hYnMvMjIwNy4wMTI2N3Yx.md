# CaTT-KWS: A Multi-stage Customized Keyword Spotting Framework based on Cascaded Transducer-Transformer

Zhanheng Yang, Sining Sun, Jin Li, Xiaoming Zhang, Xiong Wang, Long Ma, Lei Xie (2022)

## 🧩 Problem to Solve

본 논문은 엣지 디바이스(edge devices)에서 구현 가능한 맞춤형 키워드 검출(Customized Keyword Spotting, KWS) 시스템의 성능 향상을 목표로 한다. 특히, 수십 개에서 수백 개의 키워드를 동시에 처리해야 하는 실제 환경에서 발생하는 오경보(False Alarm, FA) 문제는 사용자 경험을 심각하게 저하시키는 핵심적인 문제이다. 

기존의 신경망 기반 KWS 방식은 특정 키워드 세트에 대해 높은 정확도를 보이지만, 새로운 키워드를 추가할 때마다 데이터를 다시 수집하고 모델을 재학습시켜야 하는 번거로움이 있다. 또한, 항상 켜져 있는(always-on) 시스템의 특성상 오경보를 효과적으로 제어하는 것이 매우 어려우며, 이를 해결하기 위해 기존에는 웨이크업 워드(Wake-up Word) 검출기와 커맨드 인식 모듈을 계단식으로 연결하는 방식이 사용되었다. 본 연구의 목표는 추가적인 데이터 수집이나 재학습 없이도 새로운 키워드에 유연하게 대응 가능하면서, 정확도는 유지하고 오경보는 획기적으로 줄인 맞춤형 KWS 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Cascaded Transducer-Transformer (CaTT-KWS)**라는 다단계(Multi-stage) 검증 구조를 설계하여 오경보를 단계적으로 필터링하는 것이다. 

전체 시스템은 세 단계의 파이프라인으로 구성된다. 첫 번째 단계인 **Transducer-based Detector**는 높은 재현율(Recall)을 바탕으로 키워드 후보를 빠르게 포착한다. 두 번째 단계인 **Force Alignment Module**은 스트리밍 Transducer의 고질적인 문제인 방출 지연(Emission delay)을 해결하여 키워드의 시간 경계(Time boundary)를 정밀하게 보정하고 1차 검증을 수행한다. 마지막 세 번째 단계인 **Transformer-based Decoder**는 보정된 시간 영역의 특징을 입력받아 해당 세그먼트가 실제 키워드인지 최종적으로 정밀 검증한다. 이러한 설계를 통해 높은 인식 정확도를 유지하면서도 오경보율을 극도로 낮출 수 있다.

## 📎 Related Works

최근 KWS 분야에서는 스트리밍 능력이 뛰어나고 오픈 보캐블러리(Open-vocabulary) 확장이 용이한 RNN-Transducer(RNN-T) 기반의 어쿠스틱 모델이 널리 적용되고 있다. 기존 연구들은 주로 어텐션 기반의 바이어싱(Attention-based biasing) 기법을 통해 특정 키워드의 인식률을 높이는 데 집중했으나, 다수의 키워드를 동시에 검출해야 하는 커맨드 인식 시나리오에서의 오경보 제어 문제는 충분히 다뤄지지 않았다.

또한, 오경보를 줄이기 위해 다단계 전략(Multi-stage strategy)이 제안된 바 있으며, 특히 MLD(Multi-level Detection) 방식이 대표적이다. MLD는 Transducer의 출력값만을 이용해 통계적인 신뢰도를 계산하여 검증하는 방식이다. 본 논문은 이러한 통계 기반의 검증 대신, 신경망 기반의 정밀한 시간 경계 보정과 Transformer 디코더를 통한 딥러닝 기반의 검증 단계를 도입함으로써 기존 MLD 방식보다 훨씬 낮은 오경보율을 달성하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 학습 절차
CaTT-KWS는 공유 인코더(Shared Encoder)를 기반으로 하는 다중 작업 학습(Multi-task Learning) 프레임워크이다. 인코더의 출력을 Transducer, Phone Predictor, Transformer Decoder가 공유하여 사용하며, 전체 손실 함수는 다음과 같이 세 가지 손실의 가중 합으로 정의된다.

$$L = \alpha L_{\text{Transducer}} + \beta L_{\text{CE}} + \gamma L_{\text{Transformer}}$$

여기서 $L_{\text{Transducer}}$는 Transducer 손실, $L_{\text{CE}}$는 프레임 단위 음소 예측을 위한 교차 엔트로피(Cross-Entropy) 손실, $L_{\text{Transformer}}$는 Transformer 디코더의 손실이다. 하이퍼파라미터 $\alpha, \beta, \gamma$는 각각 $1.0, 0.8, 0.5$로 설정되었다.

### 1단계: Detection Stage (Tiny Transducer)
첫 번째 단계는 DFSMN 기반 인코더와 Stateless Predictor로 구성된 Tiny Transducer와 WFST(Weighted Finite State Transducer) 디코더를 사용한다. 
- **구조**: 모델 크기를 줄여 엣지 디바이스에 최적화되었으며, 컨텍스트 독립(Context-Independent, CI) 음소를 모델링 단위로 사용한다.
- **작동**: WFST 디코딩 그래프 $LG = \min(\det(L \circ G))$를 통해 사용자가 설정한 키워드 세트 내에서 가장 가능성 높은 후보를 탐색한다. 여기서 $L$은 사전(Lexicon), $G$는 문법(Grammar)을 의미한다. 
- **역할**: 높은 재현율을 목표로 하며, 탐지된 키워드 후보의 대략적인 시간 경계를 생성한다.

### 2단계: Verification Stage (Force Alignment)
Transducer의 방출 지연 문제로 인해 1단계에서 얻은 경계는 부정확할 수 있다. 이를 보정하기 위해 Phone Predictor를 이용한 Viterbi 기반 Force Alignment를 수행한다.
- **절차**: 1단계 시작점 $t_0$에서 $t_d$ 프레임만큼 앞으로 밀어(backward) 입력 범위를 확장한다. 또한, 시작 부분의 불필요한 세그먼트를 흡수하기 위해 가비지 심볼 $(g)$을 포함한 선형 WFST 그래프를 구성한다.
- **신뢰도 계산**: 정렬 결과의 로그 확률 평균을 통해 신뢰도 점수 $S_1$을 계산한다.

$$S_1 = -\frac{\sum_{l \in f, t \in T} \log(p_l^t)}{T}$$

여기서 $p_l^t$는 $t$번째 프레임의 음소 $l$에 대한 사후 확률이며, $T$는 정밀하게 보정된 타임스탬프이다. $S_1$이 임계값 $\tau$보다 작아야 다음 단계로 진행한다.

### 3단계: Verification Stage (Transformer Decoder)
마지막 단계에서는 보정된 시간 경계 내의 인코더 출력값 $h_{\text{enc}}$를 Transformer 디코더의 입력으로 사용하여 최종 검증을 수행한다.
- **절차**: 빔 서치(Beam search)를 통해 음소 시퀀스를 생성하되, 효율성을 위해 키워드의 음소 길이 $M$만큼만 단계를 제한하여 수행한다.
- **최종 결정**: 빔 서치 결과 중 후보 키워드의 음소 시퀀스와 일치하는 결과가 있는지 확인하고, 해당 시퀀스의 점수 $S_2$를 계산한다.

$$S_2 = -\sum_{l \in B} \log(p_l)$$

여기서 $B$는 빔에서 탐지된 음소 시퀀스이다. $S_2$가 임계값 $\upsilon$보다 작을 경우 최종적으로 키워드가 트리거된 것으로 판단한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 23,000시간의 중국어 ASR 코퍼스로 학습하였으며, 29개의 중국어 커맨드를 포함한 6,000개의 발화 데이터셋(Clean 3K, Noisy 3K)으로 평가하였다. 오경보 측정에는 84시간의 배경 소음 데이터셋을 사용하였다.
- **입력 특징**: 40차원 PNCC(Power-Normalized Cepstral Coefficients)를 사용하였다.
- **모델 크기**: 전체 모델 크기는 약 3.8M 파라미터로 엣지 디바이스 배포에 적합하다.

### 주요 결과
- **오경보 감소 효과**: 단일 Transducer 모델(S0)의 오경보율은 시간당 $1.47$회였으나, 제안된 3단계 전체 프레임워크(S3)를 적용했을 때 **시간당 $0.13$회**로 획기적으로 감소하였다(90% 이상의 상대적 감소).
- **인식 정확도**: 오경보를 대폭 줄였음에도 불구하고, 인식 정확도의 하락은 $2\%$ 미만으로 매우 적었다.
- **MLD와의 비교**: ROC 커브 분석 결과, 특히 오경보율이 $0.15$회/시간 이하인 저-FA 영역에서 CaTT-KWS가 기존 MLD 방식보다 훨씬 낮은 오거부율(False Rejection Rate, FRR)을 보이며 우수한 성능을 입증하였다.
- **시간 경계 보정의 중요성**: Force Alignment를 적용한 경우(S3)가 적용하지 않은 경우(S1)보다 성능이 높았으며, 실제 시작점 오차(Start point error)가 Transducer 단독(Clean 0.29s)에서 Force Alignment 적용 후(Clean 0.11s)로 크게 줄어들었음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 다단계 검증 구조가 KWS 시스템에서 오경보를 줄이는 데 매우 효과적임을 보여주었다. 특히, 단순히 모델을 깊게 쌓는 것이 아니라 **[탐지 $\rightarrow$ 정밀 경계 보정 $\rightarrow$ 최종 검증]**으로 이어지는 논리적 파이프라인을 구축한 것이 주효했다.

**강점 및 분석**:
1. **시간 경계의 정밀도**: Transducer의 고유한 문제인 emission delay를 Force Alignment 단계에서 해결함으로써, 후속 단계인 Transformer 디코더가 정확한 특징 벡터를 입력받을 수 있게 한 점이 성능 향상의 핵심이다.
2. **모델 효율성**: 3.8M라는 작은 크기로도 고성능을 달성하여 실용성이 높다.

**한계 및 논의**:
1. **Transformer 크기의 영향**: 실험 결과, Transformer 디코더의 레이어 수를 늘리는 것이 반드시 성능 향상으로 이어지지 않았으며, 오히려 과적합(Overfitting)으로 인해 저-FA 영역에서 성능이 저하되는 현상이 발견되었다. 이는 KWS와 같은 특수 목적 모델에서는 모델의 용량보다 데이터의 특성과 검증 로직이 더 중요함을 시사한다.
2. **모델링 단위의 트레이드-오프**: 음소(Phone) 단위 모델은 오경보 억제에 유리하고, 문자(Character) 단위 모델은 오거부율을 낮추는 데 유리한 특성을 보였다. 이는 적용 서비스의 성격(보안 중심인지, 편의성 중심인지)에 따라 모델링 단위를 선택적으로 적용해야 함을 의미한다.

## 📌 TL;DR

본 논문은 엣지 디바이스용 맞춤형 키워드 검출 시스템에서 발생하는 오경보 문제를 해결하기 위해 **Transducer-Force Alignment-Transformer**로 이어지는 3단계 검증 프레임워크(**CaTT-KWS**)를 제안한다. 이 시스템은 높은 재현율로 후보를 찾고, 정밀한 시간 경계 보정을 거쳐, 최종적으로 Transformer가 검증하는 구조를 가진다. 실험 결과, 인식 정확도의 손실을 최소화하면서 오경보율을 시간당 $1.47$회에서 $0.13$회로 $90\%$ 이상 감소시키는 성과를 거두었으며, 이는 향후 저전력/고신뢰성 음성 명령 인식 시스템 구현에 중요한 기반이 될 것으로 보인다.