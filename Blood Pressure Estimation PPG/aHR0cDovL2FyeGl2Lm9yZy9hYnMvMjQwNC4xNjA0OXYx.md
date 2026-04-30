# EXPLORING THE LIMITATIONS OF BLOOD PRESSURE ESTIMATION USING THE PHOTOPLETHYSMOGRAPHY SIGNAL

Felipe M. Dias, Diego A.C. Cardenas, Marcelo A.F. Toledo, Filipe A.C. Oliveira, Estela Ribeiro, Jose E. Krieger, Marco A. Gutierrez (2024)

## 🧩 Problem to Solve

본 연구는 비침습적이고 연속적인 혈압(Blood Pressure, BP) 모니터링을 위해 광전용적맥파(Photoplethysmography, PPG) 신호를 활용하는 방법의 한계와 가능성을 분석하는 것을 목표로 한다. 고혈압은 심혈관 질환의 주요 원인으로 정밀한 혈압 관리가 필수적이지만, 기존의 커프(Cuff) 기반 방식은 간헐적 측정만 가능하며 사용자에게 불편함을 준다. 반면 침습적 동맥혈압(Invasive Arterial Blood Pressure, IABP) 측정은 정확하고 연속적이지만, 감염 및 혈전 등의 위험이 있어 제한적으로 사용된다.

최근 PPG를 이용한 혈압 추정 연구가 활발히 진행되고 있으나, 추정 정밀도에 대한 논쟁이 지속되고 있다. 특히 기존의 많은 연구가 데이터 누수(Data Leakage) 문제로 인해 성능이 과대평가되었다는 비판이 제기되었다. 따라서 본 논문은 PPG 신호가 실제로 혈압을 정확히 예측하기에 충분한 정보를 포함하고 있는지, 그리고 그 성능의 상한선(Upper Bound)은 어디까지인지를 객관적으로 검증하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **정규화된 IABP(N-IABP) 신호를 벤치마크로 활용**하여 PPG 기반 혈압 추정의 이론적 성능 한계를 설정하는 것이다. IABP 신호는 신체 내부의 압력 센서로부터 직접 측정되므로 PPG보다 더 정확한 정보를 제공한다. 여기서 IABP 신호를 Max-Min 정규화하여 절대적인 수치(Systolic/Diastolic 값)를 제거하고 형태학적(Morphological) 정보만 남긴 N-IABP를 구성한다. 

만약 직접적인 압력 정보가 제거된 N-IABP 신호만으로도 혈압 추정이 어렵다면, 간접적인 광학 측정치인 PPG 신호를 이용한 혈압 추정은 더욱 어려울 것이라는 가설을 세운다. 이를 통해 PPG 기반 혈압 추정 모델이 도달할 수 있는 현실적인 성능 지표를 제시하고, 필터링 조건에 따른 성능 변화를 분석하여 PPG 신호의 정보 손실 가능성을 탐색한다.

## 📎 Related Works

혈압 추정 방법론은 크게 두 가지로 나뉜다. 첫째는 맥파 전달 속도(Pulse Wave Velocity, PWV) 기반 접근법으로, PTT(Pulse Transit Time)나 PAT(Pulse Arrival Time)를 측정하여 이론적 방정식(Bramwell-Hill 등)을 통해 혈압을 계산한다. 이는 보정(Calibration) 과정이 필수적이며 여러 측정 지점의 신호나 ECG 신호가 동시에 필요하다는 단점이 있다. 둘째는 맥파 분석(Pulse Wave Analysis, PWA) 기반 접근법으로, 단일 지점의 PPG 파형 특징을 추출하여 머신러닝이나 딥러닝으로 혈압을 회귀 예측한다.

최근의 딥러닝 기반 연구들은 보정이 필요 없는(Calibration-free) 모델을 지향하며 높은 정확도를 보고하고 있으나, 본 논문은 이러한 연구들이 MIMIC-II-UCI 데이터셋을 사용할 때 환자 식별 정보 부재로 인한 데이터 누수 문제를 겪어 성능이 인위적으로 부풀려졌음을 지적한다. 반면 보정 기반(Calibration-based) 접근법은 초기 기준점 설정이 필요하지만 더 현실적인 예측 가능성을 제공한다.

## 🛠️ Methodology

### 전체 파이프라인
본 연구는 보정 기반의 딥러닝 파이프라인을 제안한다. 시스템은 (1) 보정 시점의 신호(N-PPG 또는 N-IABP)와 해당 시점의 혈압 값(SBP, DBP), (2) 추론 대상이 되는 시점의 신호(N-PPG 또는 N-IABP)를 입력으로 받아 대상 신호의 혈압을 예측한다.

### 데이터 전처리 및 정규화
- **데이터셋**: VitalDB 데이터셋을 사용하여 PPG와 IABP가 모두 기록된 3,338명의 환자 데이터를 추출하였다.
- **윈도우 및 정렬**: 신호를 24초(2400 samples, 100Hz) 단위로 분할하고, 상호 상관(Cross-correlation)을 통해 PPG와 IABP 간의 시간 지연(Lag)을 보정하여 정렬하였다.
- **신호 품질 분석**: 4차 Chebyshev Type II 밴드패스 필터를 적용한 후, 평균 비트(Mean beat)와 개별 비트 간의 피어슨 상관계수가 0.9 이상인 윈도우만 유효한 데이터로 간주하였다.
- **정규화(Normalization)**: Max-Min 정규화를 통해 신호의 절대적 진폭(SBP, DBP 값)을 제거하고 파형의 형태만 남긴 N-IABP와 N-PPG를 생성하였다.
- **페어 몽타주(Pair Montage)**: 보정 신호와 추론 신호 쌍을 구성할 때, 시간 간격을 3분에서 2시간 사이로 설정하고 SBP 차이가 60 mmHg 이내인 경우만 선택하여 과적합을 방지하였다.

### 모델 아키텍처: Siamese ResNet
모델은 두 개의 동일한 ResNet 기반 네트워크가 공유 가중치를 사용하는 Siamese 구조로 설계되었다.

1. **Base Network**: 
   - 1D Convolution $\rightarrow$ Batch Normalization $\rightarrow$ ReLU 순으로 처리한다.
   - 이후 4개의 ResNet-like 블록이 이어지며, 각 블록은 필터 수 128, 196, 256, 320개를 사용한다.
   - 각 블록 내부에서는 Stride $S=4$인 컨볼루션 경로와 Max Pooling 및 $1 \times 1$ 컨볼루션을 통한 지름길(Shortcut) 경로가 합쳐지며, ReLU와 Dropout(20%)이 적용된다.
   - 마지막으로 Global Average Pooling을 통해 특성 벡터를 추출한다.

2. **Fusion and Prediction**:
   - 보정 신호의 특성 벡터, 추론 신호의 특성 벡터, 그리고 보정 시점의 SBP/DBP 수치를 모두 결합(Concatenate)한다.
   - 결합된 벡터는 세 개의 Fully-connected 레이어(128 $\rightarrow$ 64 $\rightarrow$ 2)를 통과하여 최종적으로 $\text{SBP}$와 $\text{DBP}$를 출력한다.

### 학습 및 비교 대상
- **손실 함수**: 회귀 분석을 위한 선형 활성화 함수를 출력층에 사용하였다.
- **베이스라인 모델**: 추론 혈압을 단순히 보정 혈압과 동일하게 예측하는 모델($P_{\text{inference}} = P_{\text{calibration}}$)을 설정하여, 모델이 단순 복제가 아닌 실제 상관관계를 학습하는지 확인하였다.

## 📊 Results

### 실험 설정
- **지표**: AAMI 표준(평균 차이 $\pm 5\text{ mmHg}$, 표준편차 $\le 8\text{ mmHg}$), BHS 표준(오차 범위 $\le 5, 10, 15\text{ mmHg}$에 따른 A~D 등급), MAE, 피어슨 상관계수($\rho$)를 사용하였다.
- **필터 조건**: 필터 없음(Raw), $0.5\text{ Hz} \sim 10\text{ Hz}$, $0.5\text{ Hz} \sim 3.5\text{ Hz}$ 세 가지 조건을 비교하였다.

### 주요 결과
1. **N-IABP 성능**: 
   - Raw 신호의 경우 SBP와 DBP 모두 AAMI 표준을 만족하였으며, BHS 평가에서 **Grade A**를 획득하였다.
   - 필터 범위가 좁아질수록($0.5 \sim 3.5\text{ Hz}$) 성능이 크게 하락하여 SBP는 Grade D까지 떨어졌다.

2. **N-PPG 성능**:
   - DBP의 경우 BHS Grade B를 기록하였으나, SBP는 모든 조건에서 **Grade D**를 기록하며 AAMI 표준을 만족하지 못했다.
   - Raw 신호와 $0.5 \sim 10\text{ Hz}$ 필터 신호 간의 성능 차이는 미미했으나, $0.5 \sim 3.5\text{ Hz}$ 필터 적용 시 성능이 급격히 저하되었다.

3. **비교 분석**:
   - N-IABP 모델이 N-PPG 모델보다 모든 지표에서 일관되게 우수한 성능을 보였다.
   - 다만, 두 모델 모두 베이스라인 모델보다는 높은 성능을 보여, 단순 보정값 복제가 아닌 신호의 형태학적 정보를 활용해 혈압을 추정하고 있음을 증명하였다.

## 🧠 Insights & Discussion

본 연구는 N-IABP라는 성능 상한선을 설정함으로써 PPG 기반 혈압 추정의 현실적인 한계를 명확히 제시하였다. N-IABP 신호가 SBP/DBP 수치 정보 없이 형태만으로도 AAMI 표준을 만족한다는 점은, 이론적으로 신호의 형태학적 정보 내에 혈압 관련 정보가 충분히 존재함을 시사한다.

그러나 N-PPG의 성능이 N-IABP에 비해 현저히 낮게 나타난 점은 주목할 만하다. 이는 PPG 신호가 IABP 신호의 필터링된 형태라는 가설을 뒷받침하며, 이 과정에서 혈압 추정에 필수적인 고주파 성분이나 세부 정보가 손실되었을 가능성을 보여준다. 특히 $0.5 \sim 3.5\text{ Hz}$ (심박수 범위) 필터 적용 시 성능이 급락하는 결과는, 혈압 정보가 단순한 심박수 정보 이상의 고차원적 파형 특성에 포함되어 있음을 의미한다.

결론적으로 PPG 신호는 혈압과 상관관계가 있는 정보를 일부 포함하고는 있으나, 현재의 딥러닝 접근법만으로는 정밀한 혈압 추정을 위한 충분한 정보량을 제공하지 못할 수 있다. 이는 향후 소비자 가전용 웨어러블 기기에서 PPG를 통한 혈압 측정 기능을 구현할 때 매우 신중한 접근과 추가적인 정보(예: 다른 생체 신호의 결합)가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 정규화된 IABP(N-IABP) 신호를 이용한 혈압 추정 모델을 구축하여, PPG 기반 혈압 추정의 성능 상한선을 정의하였다. 실험 결과, N-IABP는 높은 정확도(BHS Grade A)를 보였으나 N-PPG는 SBP 추정에서 매우 낮은 성능(Grade D)을 보여, PPG 신호만으로는 정밀한 혈압 예측에 한계가 있음을 입증하였다. 이 연구는 PPG 기반 혈압 추정 분야의 과도한 낙관론에 경종을 울리며, 현실적인 성능 벤치마크를 제공함으로써 향후 연구의 방향성을 제시한다.