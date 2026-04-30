# Privacy-Preserving and Trustworthy Deep Learning for Medical Imaging

Kiarash Sedghighadikolaei, Attila A Yavuz (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석을 자동화하고 효율화하는 **Deep Radiomics**(딥러닝 기반의 라디오믹스) 과정에서 발생하는 데이터 프라이버시 및 신뢰성 문제를 해결하고자 한다. 

의료 데이터는 매우 민감한 개인정보를 포함하고 있으며, 최근의 Deep Radiomics는 고성능 연산 자원을 필요로 하기 때문에 클라우드 서비스로의 아웃소싱이 빈번하게 이루어진다. 그러나 이 과정에서 클라우드 데이터 유출, 법적 규제 위반, 악성 소프트웨어 감염 등의 보안 위협이 상존한다. 

기존의 프라이버시 강화 기술(Privacy-Enhancing Technologies, PETs)에 관한 연구들은 대개 일반적인 머신러닝(ML) 알고리즘에 집중하거나, 너무 광범위한 개요만을 제공하는 경향이 있었다. 특히 Deep Radiomics 파이프라인(데이터 생성 $\rightarrow$ 수집 $\rightarrow$ 학습 $\rightarrow$ 추론) 전체에 걸쳐 효율성, 정확도, 프라이버시를 동시에 충족하며 실무적으로 적용 가능한 통합 방안에 대한 체계적인 분석이 부족한 상황이다. 따라서 본 논문의 목표는 PETs를 Deep Radiomics 파이프라인에 효과적으로 통합하기 위한 분류 체계(Taxonomy)를 제시하고, 실무적인 하이브리드 설계 방안과 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Deep Radiomics를 위한 **Privacy-preserving Deep Radiomics (PPDR)** 프레임워크를 체계화한 것에 있으며, 세부 내용은 다음과 같다.

1.  **PETs의 상세 분류 및 하이브리드 구성 제시**: Deep Radiomics에 적용 가능한 PETs를 기본 원리에 따라 4가지 그룹으로 분류하고, 단일 기술의 한계를 극복하기 위한 실무적인 하이브리드 설계(Hybrid Designs) 방안을 제시하였다.
2.  **Deep Radiomics 파이프라인 단계별 기술적 통찰 제공**: 데이터 저장 및 검색, 모델 학습, 모델 배포 및 추론이라는 전체 파이프라인의 각 단계에서 각 PET가 어떻게 통합될 수 있는지, 그리고 이때 발생하는 성능 저하 및 프라이버시 공격에 대한 대응 방안을 분석하였다.
3.  **PPDR을 위한 로드맵 및 향후 연구 방향 제안**: 각 PET의 기능적 한계와 통합 과정의 도전 과제를 식별하고, 이를 해결하기 위한 미래 연구 방향을 제시하여 실질적인 구현을 위한 가이드라인을 제공하였다.

## 📎 Related Works

논문에서는 기존의 PPML(Privacy-preserving Machine Learning) 및 PETs 관련 연구들을 다음과 같이 검토한다.

-   **암호학적 접근**: Searchable Encryption(SE), Private Information Retrieval(PIR), Oblivious RAM(ORAM), Multi-Party Computation(MPC), Homomorphic Encryption(HE), Functional Encryption(FE) 등이 연구되어 왔으며, 특히 HE와 MPC가 PPDL(Privacy-preserving Deep Learning) 분야에서 주목받았다.
-   **검증 및 하드웨어 접근**: Zero-Knowledge Proofs(ZKP)를 통한 실행 검증과 Trusted Execution Environments(TEEs)를 통한 하드웨어 기반 격리 실행 등이 제안되었다.
-   **학습 패러다임 및 통계적 접근**: 데이터 공유 없이 학습하는 Federated Learning(FL)과 노이즈를 추가하여 통계적 프라이버시를 보장하는 Differential Privacy(DP) 등이 널리 연구되었다.

**기존 연구와의 차별점**: 기존 연구들은 특정 PET의 기술적 세부 사항이나 일반적인 ML 적용 사례에 치중했다. 반면, 본 연구는 **Deep Radiomics의 특수성**(고차원 의료 영상 데이터, CNN 아키텍처의 비선형성, 정밀 의료를 위한 고정확도 요구 등)을 고려하여 전체 파이프라인 관점에서 PETs의 실무적 통합 가능성과 최적의 조합을 분석했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 Deep Radiomics 파이프라인의 각 단계에 적용 가능한 PETs를 다음과 같이 체계화하여 설명한다.

### 1. 데이터 저장, 액세스 및 검색 (Data Storage, Access, and Retrieval)
데이터가 클라우드에 저장된 상태에서 프라이버시를 유지하며 필요한 데이터를 찾는 단계이다.
-   **Searchable Encryption (SE)**: 데이터를 암호화하여 저장하고, 키워드 기반의 검색 토큰을 통해 복호화 없이 검색을 수행한다. 동적 업데이트가 가능한 DSSE(Dynamic SE)가 효율적이지만, 검색 패턴을 통한 정보 유출 위험이 있다.
-   **Private Information Retrieval (PIR)**: 서버가 사용자가 어떤 인덱스의 데이터를 요청했는지 알 수 없게 한다. 정보 이론적(IT-PIR) 방식과 계산적(CPIR) 방식으로 나뉜다.
-   **Oblivious RAM (ORAM)**: 메모리 액세스 패턴 자체를 숨겨 SE나 PIR에서 발생하는 패턴 유출 문제를 해결한다.

### 2. 암호화 상태에서의 실행 (Execution Under Encryption)
데이터를 복호화하지 않고 학습 및 추론을 수행하는 핵심 단계이다.
-   **Secure Multi-party Computation (MPC)**: 여러 당사자가 입력값을 숨긴 채 공동으로 함수를 계산한다.
    -   **Garbled Circuits (GC)**: 함수를 논리 게이트 회로로 표현하여 계산하며, 비선형 연산에 유리하다.
    -   **Secret Sharing (SS)**: 데이터를 조각내어 분산 저장하고, 이를 통해 선형 연산(덧셈, 곱셈)을 효율적으로 수행한다.
    -   **Oblivious Transfer (OT)**: 송신자가 보낸 여러 값 중 수신자가 선택한 값만 전송받으며, 송신자는 무엇이 선택되었는지 알 수 없다.
-   **Homomorphic Encryption (HE)**: 암호문 상태에서 연산을 수행하면 결과물 또한 암호문이며, 이를 복호화하면 평문 연산 결과와 동일하다. 
    -   선형 연산은 효율적이나, ReLU와 같은 비선형 활성화 함수는 다항식 근사(Polynomial Approximation)를 사용해야 하므로 정확도 손실이 발생할 수 있다.
-   **Functional Encryption (FE)**: 특정 함수에 대한 결과값만을 복호화할 수 있는 특수 키를 제공하여, 데이터 전체가 아닌 필요한 정보만 선택적으로 추출한다.

### 3. 실행 검증 (Verifiable Execution)
계산 결과가 정당하게 산출되었는지 확인하는 단계이다.
-   **Zero-Knowledge Proof (ZKP)**: 증명자(Prover)가 비밀 정보를 공개하지 않고도 특정 문장이 참임을 검증자(Verifier)에게 입증한다. 특히 **zkSNARKs**는 증명 크기가 매우 작고 검증 속도가 빨라 Deep Radiomics의 모델 소유권 확인 및 추론 검증에 유용하다.

### 4. 협력 학습 패러다임 (Collaborative Learning)
-   **Federated Learning (FL)**: 데이터를 중앙으로 모으지 않고, 각 로컬 노드에서 학습된 모델 파라미터(Gradient)만을 전송하여 글로벌 모델을 업데이트한다. 

### 5. 신뢰 실행 환경 (Trusted Hardware)
-   **Trusted Execution Environments (TEEs)**: CPU 내부에 격리된 보안 영역인 **Secure Enclave**(예: Intel SGX)를 구축하여, 운영체제나 하이퍼바이저조차 접근할 수 없는 안전한 영역에서 연산을 수행한다. 암호학적 방식보다 속도가 월등히 빠르다.

### 6. 통계적 섭동 (Statistical Perturbation)
-   **Differential Privacy (DP)**: 데이터셋에 무작위 노이즈를 추가하여 특정 개별 데이터의 포함 여부를 알 수 없게 만든다. 
    -   **Local DP (LDP)**: 사용자가 데이터를 보내기 전 직접 노이즈를 추가한다.
    -   **Central DP (CDP)**: 신뢰할 수 있는 중앙 관리자가 수집 후 노이즈를 추가한다.

## 📊 Results

본 논문은 새로운 알고리즘을 제안하는 실험 논문이 아니라, 기존 PETs를 체계적으로 분석한 **Systematization/Survey 논문**이다. 따라서 정량적인 벤치마크 결과 대신, 다양한 PETs의 특성을 비교 분석한 결과(Table I 참조)를 제시한다.

### 주요 비교 분석 결과
-   **계산 및 통신 비용**: 
    -   **TEEs**가 가장 낮은 계산 비용과 통신 비용을 보이며 실무 적용 가능성이 가장 높다.
    -   **HE**와 **MPC**는 보안성은 매우 높으나 계산 오버헤드와 통신 횟수가 극심하여 실시간 적용에 한계가 있다.
-   **정확도 (Accuracy)**: 
    -   **HE**나 **DP**는 비선형 함수 근사 및 노이즈 추가로 인해 모델의 정확도가 저하되는 trade-off가 발생한다.
    -   **TEEs**와 **MPC**는 원본 연산을 그대로 수행하므로 정확도 손실이 거의 없다.
-   **보안성**: 
    -   **HE**는 수학적 증명 기반의 강력한 보안을 제공한다.
    -   **TEEs**는 하드웨어 의존적이므로 사이드 채널 공격(Side-channel attack)에 취약할 수 있다.

## 🧠 Insights & Discussion

### 강점 및 기회
Deep Radiomics의 보안을 위해서는 단일 PETs보다는 **하이브리드 설계**가 필수적이다. 예를 들어, 선형 연산은 **HE**로 처리하고 비선형 연산(ReLU 등)은 **MPC**나 **TEEs**로 처리함으로써 보안성과 효율성을 동시에 잡을 수 있다. 또한, TEEs와 ORAM을 결합하면 하드웨어 수준에서 액세스 패턴 유출까지 방지할 수 있다.

### 한계 및 미해결 과제
1.  **비선형성 처리**: CNN의 핵심인 활성화 함수와 풀링 계층을 암호화 상태에서 효율적으로 처리하는 방법은 여전히 난제이다.
2.  **정확도-프라이버시 트레이드오프**: DP의 노이즈 레벨을 높이면 프라이버시는 강화되나, 정밀 의료에서 요구하는 고정확도 진단 성능이 하락하는 문제가 있다.
3.  **하드웨어 가속**: 대부분의 암호학적 PETs가 GPU 가속을 지원하지 않아 대규모 의료 영상 데이터셋 처리 시 심각한 지연 시간이 발생한다.

### 비판적 해석
본 논문은 매우 포괄적인 분류 체계를 제공하지만, 구체적인 수치 기반의 성능 비교 데이터(예: 특정 CNN 모델에서의 추론 시간 비교)가 부족하여 실제 구현 시 어떤 조합이 최적인지 결정하기에는 정성적인 분석에 치중되어 있다는 아쉬움이 있다.

## 📌 TL;DR

본 논문은 의료 영상 분석을 위한 **Deep Radiomics 파이프라인 전체에 걸쳐 프라이버시를 보호하기 위한 PETs의 통합 체계(PPDR)**를 제안한다. 암호학적 방법(MPC, HE, FE), 검증 방법(ZKP), 하드웨어 방법(TEE), 학습 패러다임(FL), 통계적 방법(DP)을 체계적으로 분류하고, 이들의 장단점 및 하이브리드 구성 방안을 분석하였다. 이 연구는 향후 보안성과 효율성을 동시에 갖춘 의료 AI 시스템을 구축하기 위한 설계 도면(Roadmap)으로서 중요한 역할을 할 것으로 기대된다.