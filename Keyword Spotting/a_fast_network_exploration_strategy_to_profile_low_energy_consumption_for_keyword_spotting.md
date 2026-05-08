# A Fast Network Exploration Strategy to Profile Low Energy Consumption for Keyword Spotting

Arnab Neelim Mazumder, Tinoosh Mohsenin (2022)

## 🧩 Problem to Solve

본 논문은 스마트 기기의 음성 기반 사용자 인터랙션을 위한 Keyword Spotting(KWS) 시스템의 에너지 효율성 문제를 해결하고자 한다. KWS는 특정 키워드(예: 'Hey Siri', 'Ok Google')를 감지하여 메인 시스템을 활성화하는 역할을 하므로, 전력 소모를 줄이기 위해 항상 켜져(always-on) 있어야 한다. 하지만 이러한 특성 때문에 배터리 수명에 큰 부담을 주게 된다.

특히, 제한된 자원을 가진 FPGA나 마이크로컨트롤러와 같은 엣지 디바이스에 딥러닝 모델을 배포할 때, 요구되는 정확도를 유지하면서 하드웨어 제약 조건(전력, 메모리)을 충족하는 최적의 네트워크 구성(configuration)을 찾는 것은 매우 어려운 과제이다. 따라서 본 연구의 목표는 하드웨어 인지적(hardware-aware) 방식으로 정확도와 에너지 소비 사이의 최적의 트레이드-오프를 제공하는 네트워크 파라미터를 빠르게 탐색하는 회귀 기반(regression-based) 기법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 네트워크의 필터 스케일링(scaling, $s$)과 양자화(quantization, $q$)라는 두 가지 주요 변수를 조절하여 하드웨어 효율성을 최적화하는 것이다. 이를 위해 모든 조합을 실제로 구현하여 테스트하는 대신, 소수의 샘플 데이터를 통해 정확도와 에너지 소비를 예측하는 회귀 모델을 구축함으로써 탐색 시간을 획기적으로 단축하였다.

주요 기여 사항은 다음과 같다:

- KWS를 위한 DNN의 에너지 소비를 프로파일링하기 위해 실험적 및 분석적 방법론에 기반한 체계적인 접근 방식을 도입하였다.
- 양자화된 $\text{NN}\langle q, s \rangle$ 모델의 정확도와 하드웨어 에너지 소비를 예측하는 두 가지 다항 회귀(polynomial regression) 설정을 제시하였다.
- 다양한 연산 엔진(Processing Engines, PEs) 수와 정밀도 수준을 지원하며, 저전력·저비용 FPGA에서 구현 가능한 파라미터화 및 확장 가능한 하드웨어 가속기를 설계하였다.

## 📎 Related Works

KWS는 전통적으로 CNN을 통해 구현되어 왔으며, 최근에는 BiLSTM이나 CRNN과 같은 대안적 구조도 제안되었다. 오디오 신호 처리 단계에서는 raw audio를 직접 처리하는 sinc convolutions 방식이 등장했으나, 여전히 MFCC(Mel-frequency cepstrum)를 이용한 특징 추출 방식이 널리 사용되고 있다.

모델 압축 측면에서는 BNN(Binarized Neural Networks)이나 TNN(Ternarized Neural Networks)과 같은 극단적인 양자화 기법들이 연구되었다. 또한, Neural Architecture Search(NAS)를 통해 최적의 정확도를 갖는 셀 기반 구조를 찾는 연구들이 진행되었으나, 이러한 방식들은 실제 하드웨어 배포 관점에서의 에너지 효율성을 직접적으로 고려하지 않는 한계가 있다. 본 논문은 기존의 회귀 기반 Fast NAS 전략을 KWS와 FPGA 환경으로 확장하여 하드웨어-소프트웨어 공동 설계 관점에서 접근함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 탐색 전략

본 논문은 다음의 5단계 절차를 통해 최적의 네트워크 구성을 탐색한다.

1. baseline DNN을 사용하여 제한된 범위의 $q$와 $s$에 대해 정확도 경향성을 학습한다.
2. 학습된 데이터를 바탕으로 정확도를 예측하는 회귀 모델(approximator)을 구축한다.
3. 소수의 baseline DNN 구성을 실제 FPGA에 배포하여 에너지 소비 경향을 측정한다.
4. 측정된 데이터를 바탕으로 에너지 소비를 예측하는 회귀 모델을 구축한다.
5. 두 회귀 모델을 동시에 활용하여 정확도와 에너지 효율의 최적 트레이드-오프 지점을 선택한다.

### 2. 핵심 변수 및 모델 분석

본 연구는 에너지 소비에 가장 큰 영향을 미치는 두 가지 요소인 필터 스케일링($s$)과 양자화($q$)에 집중한다. 사용된 CNN 아키텍처의 연산량과 모델 크기는 다음과 같은 관계를 갖는다.

- 총 연산량: $\propto s^2$
- 모델 크기: $\propto q \cdot s^2$
- 최대 피처 맵 크기: $\propto q \cdot s^4$ (일부 레이어 기준)

### 3. 회귀 방정식

#### 정확도 회귀 (Accuracy Regression)

실험적으로 얻은 $(q, s, \text{accuracy})$ 데이터 포인트들을 기반으로 최소제곱법을 사용하여 다음과 같은 다항 회귀 식을 도출한다.
$$\text{Accuracy}(\text{NN}\langle q, s \rangle) \approx \hat{A}_6 \cdot q \cdot s + \hat{A}_5 \cdot s + \hat{A}_4 \cdot q + \hat{A}_3 \cdot q \cdot s + \hat{A}_2 \cdot s + \hat{A}_1 \cdot q + \hat{A}_0$$
여기서 $\hat{A}_i$는 학습된 상수 파라미터이다.

#### 에너지 회귀 (Energy Regression)

에너지는 전력 소비($\text{Power}$)와 지연 시간($\text{Latency}$)의 곱으로 정의된다.

- **전력 소비($\text{Power}$):** 메모리 통신, 곱셈, 덧셈 및 정적 전력의 합으로 모델링한다.
$$\text{Power}(\text{HW}|\text{NN}\langle q, s \rangle) \approx \hat{B}_3 \cdot q^2 \cdot s^2 + \hat{B}_2 \cdot q \cdot s^2 + \hat{B}_1 \cdot q \cdot s + \hat{B}_0$$
- **지연 시간($\text{Latency}$):** 총 연산량과 하드웨어 성능의 관계를 통해 다음과 같이 정의된다.
$$\text{Latency} \propto (s + C)$$ (여기서 $C$는 첫 번째 레이어의 지연 시간 관련 상수)
- **최종 에너지 방정식:**
$$\text{Energy}(\text{HW}|\text{NN}\langle q, s \rangle) \approx (\hat{B}_3 \cdot q^2 \cdot s^2 + \hat{B}_2 \cdot q \cdot s^2 + \hat{B}_1 \cdot q \cdot s + \hat{B}_0)(\hat{D} \cdot s + \hat{E})$$

### 4. 하드웨어 가속기 설계

Xilinx Artix-7 FPGA를 기반으로 설계된 가속기는 다음과 같은 구조를 갖는다.

- **PE Array:** $P$개의 Processing Engine으로 구성되며, 각 PE는 8개의 MAC(Multiply-Accumulate) 유닛을 가져 병렬 연산을 수행한다.
- **Memory:** Feature Map Memory, Weight Memory, Output Memory로 구분되며, 너비는 $M \cdot q$에 의해 결정된다.
- **Maxpooling Block:** Comparator를 이용한 Bubble Sort 전략으로 최대값을 추출한다.
- **병렬화 전략:** Output Channel Tiling 기법을 사용하여 각 PE가 하나의 출력 채널을 전담하게 함으로써 PE의 유휴 상태를 최소화하고 피크 성능에 근접하게 설계하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Google Speech Commands (30개 클래스, 0.5s~1s 길이).
- **전처리:** raw audio를 MFCC(Mel-frequency cepstrum)를 통해 $44 \times 13$ 크기의 스펙트로그램으로 변환.
- **하드웨어 플랫폼:** Xilinx AC 701 (Artix-7 XC7A200T FPGA), 동작 주파수 100 MHz.
- **평가 지표:** 정확도(%), 에너지 소비(mJ), 지연 시간(ms), GOPJ(Giga-Operations Per Joule).

### 2. 정량적 결과

- **회귀 모델의 정확성:** 정확도 예측 모델의 RMSE는 0.9, 에너지 예측 모델의 RMSE는 $0.01\text{mJ}$으로 매우 높은 적합도를 보였다.
- **최적 구성:** $q=4$ (4-bit), $s=4.5$ 구성에서 약 $90.1\%$의 정확도를 달성하며 에너지 효율의 최적점을 찾았다.
- **비교 분석:**
  - 기존 FPGA 기반 구현[6] 대비 에너지는 약 $2.1\times$ 감소하였고, 에너지 효율(GOPJ)은 $4\times$ 향상되었다.
  - Cortex-M7 마이크로컨트롤러 기반 구현[19] 대비 100 MHz 동작 시 약 $8.5\times$ 빠른 속도를 보였다.

## 🧠 Insights & Discussion

본 논문은 양자화 비트 수($q$)와 필터 스케일($s$) 사이의 상호보완적 관계를 명확히 제시하였다. 분석 결과, 2-bit 또는 3-bit와 같은 낮은 정밀도에서 $90\%$ 이상의 정확도를 얻으려면 네트워크 스케일($s$)을 상당히 키워야 하며, 이는 결국 전체 에너지 소비를 증가시키는 결과를 초래한다. 반면, 8-bit와 같은 고정밀도 네트워크는 스케일을 낮출 수 있음에도 불구하고 기본 연산 비용으로 인해 에너지 소비가 더 높게 나타났다.

결과적으로 **4-bit에서 5-bit 사이의 정밀도가 KWS 작업에서 에너지 효율과 정확도의 최적 균형점**임을 확인하였다. 또한, 제안된 회귀 기반 탐색 전략은 단 12개의 데이터 포인트만으로도 하드웨어 배포 전 성능을 매우 정확하게 예측할 수 있음을 입증하여, 방대한 하드웨어 실험 시간을 획기적으로 줄일 수 있음을 보여주었다.

다만, 본 연구는 필터 스케일링과 양자화라는 두 가지 변수에만 집중하였으며, 논문 서두에서 언급한 입력 해상도, 네트워크 깊이, 희소성(sparsity) 등의 다른 영향 요인들은 분석 범위에서 제외되었다는 한계가 있다.

## 📌 TL;DR

본 연구는 KWS 모델을 FPGA에 효율적으로 배포하기 위해 **필터 스케일($s$)과 양자화 비트($q$)를 변수로 하는 회귀 기반의 빠른 네트워크 탐색 전략**을 제안하였다. 소수의 샘플로 정확도와 에너지 소비를 예측하는 다항 회귀 모델을 구축하여 최적의 구성을 빠르게 찾았으며, 이를 통해 **정확도 $90.1\%$를 유지하면서 기존 연구 대비 에너지 효율을 $4\times$ 향상**시켰다. 이 방법론은 하드웨어 자원이 극히 제한된 엣지 디바이스용 AI 모델 최적화에 매우 유용한 가이드라인을 제공한다.
