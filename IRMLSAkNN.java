package moa.classifiers.multilabel;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.*;

import java.util.*;
import java.util.List;

public class IRMLSAkNN extends AbstractMultiLabelLearner implements MultiLabelClassifier {

	public void write(String fileName, String data) {
		try {
			// Verifica se o arquivo já existe
			File file = new File(fileName);
			FileWriter writer;
			if (file.exists() && !file.isDirectory()) {
				writer = new FileWriter(fileName, true); // Adiciona ao final do arquivo
			} else {
				writer = new FileWriter(fileName); // Cria um novo arquivo
			}

			// Escreve a string no arquivo
			writer.write(data);

			// Fecha o escritor
			writer.close();

		} catch (IOException e) {
			System.err.println("Erro ao escrever no arquivo " + fileName + ": " + e.getMessage());
		}
	}

	private static Random random = new Random();
	private static final long serialVersionUID = 1L;

	public static int contatorInstancias;
	//Tamanho maximo da janela a ser utilizada
	public IntOption maxWindowSize = new IntOption("maxWindowSize", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

	//Tamanho minimo da janela
	public IntOption minWindowSize = new IntOption("minWindowSize", 'm', "The minimum number of instances to sotre",   50, 1, Integer.MAX_VALUE);

	//K para o MLSMOTE
	public IntOption mlsmoteK = new IntOption("mlsmoteK", 'q', "The number of k for MLSMOTE",   3, 1, Integer.MAX_VALUE);

	//MLSMOTE POSITION
	private String[] balancingPositions = { "DEFAULT", "FIRST", "SECOND", "THIRD"};

	public MultiChoiceOption balancingPosition = new MultiChoiceOption("balancingPosition", 'b', "The position of the balancing method", balancingPositions, balancingPositions, 0);

	//Frequencia a aplicar o metodo de balanceamento
	public IntOption balancingFrequency = new IntOption("balancingFrequency", 'g', "Number of iterations to balance",   1, 0, Integer.MAX_VALUE);

	//TIPO DE GERACAO DE ROTULOS
	private String[] labelGenerationMethods = { "INTERSECTION", "UNION", "RANKING"};

	public MultiChoiceOption labelGenerationMethod = new MultiChoiceOption("labelGenerationMethod", 'l', "The method for label generation", labelGenerationMethods, labelGenerationMethods, 0);

	private String[] instanceTypes = { "SPARSE", "DENSE"};

	public MultiChoiceOption instanceType = new MultiChoiceOption("instanceTypes", 'c', "The method for label generation", instanceTypes, instanceTypes, 0);


	//Campo privado que contem as possiveis metricas de avaliacao a serem utilizadas
	private String[] punishmentTypes = {"Static", "NO-PUNISHMENT", "Dynamic", "G-PUNISHMENT"};

	//Metodo de balanceamento a ser utilizado
	public MultiChoiceOption punishmentType = new MultiChoiceOption("punishmentType", 'h', "Choose the method used to balance the window.", punishmentTypes, punishmentTypes, 0);

	//Campo privado que contem as possiveis metricas de avaliacao a serem utilizadas
	private String[] balancingMethods = {"DEFAULT", "MLSMOTE", "MLROS", "MLRUS"};

	//Metodo de balanceamento a ser utilizado
	public MultiChoiceOption balancingMethod = new MultiChoiceOption("balancingMethod", 'f', "Choose the method used to balance the window.", balancingMethods, balancingMethods, 0);

	//Taxa de penalidade a ser aplicada
	public FloatOption penalty = new FloatOption("penalty", 'p', "Penalty ratio", 1, 0, Float.MAX_VALUE);

	//Taxa de incremento da classe minoritaria
	public FloatOption minorityIncrement = new FloatOption("majorityIncrement", 'i', "Percentage of size increment. Use this option for MLROS", 1, 0, Float.MAX_VALUE);

	//Taxa de reducao da classe majoritaria
	public FloatOption majorityReduction = new FloatOption("majorityReduction", 'y', "Percentage of size reduction. Use this option for MLRUS", 1, 0, Float.MAX_VALUE);

	//Taxa de reducao
	public FloatOption reductionRatio = new FloatOption("reductionRatio", 'r', "Reduction ratio", 0.5, 0, 1);

	//Campo privado que contem as possiveis metricas de avaliacao a serem utilizadas
	private String[] metrics = {"Subset Accuracy", "Hamming Score", "F-Score", "G-Mean"};

	//Seletor para metricas de avaliacao a serem utilizadas
	public MultiChoiceOption metric = new MultiChoiceOption("metric", 'e', "Choose metric used to adjust memory", metrics, metrics, 0);

	//Input para o tamanho do historico de k
	public IntOption kHistorySize = new IntOption("kHistorySize", 'k', "The history length for determining K value", 100, 1, Integer.MAX_VALUE);
	private String[] distances = {"MANHATTAN", "EUCLIDEAN"};

	//Seletor para metricas de avaliacao a serem utilizadas
	public MultiChoiceOption distance = new MultiChoiceOption("distanceType", 'd', "Choose metric used to adjust memory", distances, distances, 1);

	//Quantidade de rotulos
	private int numLabels;

	//K atual
	private int[] currentK;

	//Estrutura de dados para armazenar o historico da metrica K.
	private List<Integer>[][] KmetricHistory;

	//Estrutura de dados para armezanar as instancias utilizadas na janela atual
	private List<Instance> window;

	//Matriz com as distancias entre os vizinhos pré-computadas
	private double[][] distanceMatrix;


	private HashMap<Object, Object> distanceMatrix2;

	//Numero minimo de atributos a serem utilizados
	private double[] attributeRangeMin;

	//Numero maximo de atributos a serem utilizados
	private double[] attributeRangeMax;

	//Vetor WxL que mapeia quais rotulos estao habilitados para cada instancia na janela
	private int[][] labelInstanceMask;

	//Historico de predicoes realizadas
	private Map<Integer, List<Double>> predictionHistories;
	private Map<Integer, List<int[]>> predictionHistoriesAdapted;
	//Historico
	private Map<Instance, Integer> errors;

	private double[] labelIR;
	private Map<Instance, Double> instanceMeanIR;
	private double windowMeanIR;
	private double windowMaxIR;
	private int windowPresentLabels;

	private double globalMeanIR;
	private double globalMaxIR;
	private double[][] globalLabelIR;
	private double globalMaxCount;

	private double windowMaxCount;

	private String dadosTreino = "";

	public String getDadosTreino(){
		return this.dadosTreino;
	}


	public void setDadosTreino(String s){
		this.dadosTreino = s;
	}

	private int nInstancia = 0;

	public int getnInstancia(){
		return this.nInstancia;
	}


	protected Instances multilabelStreamTemplate = null;

	@Override
	public String getPurposeString() {
		return "Self-Adjusting k Nearest Neighbors for Multi-Label Drifting Data Streams";
	}


	public static int generateRandomNumber(double lambda) {

		int x = 0;
		double p = Math.exp(-lambda);
		double F = p;

		double u = random.nextDouble();
		while (u > F) {
			p = (lambda * p) / (x + 1);
			F += p;
			x++;
		}
		return x;
	}

	//Metodo para capturar as informações da tela e inicializar o modelo
	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			//Quantidade de rotulos
			numLabels = context.numOutputAttributes();

			//janela de dados
			window = new ArrayList<Instance>();

			//Quantidade minima de atributos a serem considerados
			attributeRangeMin = new double[context.numInputAttributes()];

			//Quantidade maxima de atributos a serem considerados
			attributeRangeMax = new double[context.numInputAttributes()];

			//Matriz de distancias
			distanceMatrix = new double[maxWindowSize.getValue()][maxWindowSize.getValue()];

			//Historico de predicoes
			predictionHistories = new HashMap<Integer, List<Double>>();

			//Historico e erros
			errors = new HashMap<Instance, Integer>();

			//Inicializa a mascara de rotulos habilitados por instancia
//			System.out.println("o tamanho maximo da janela eh: " + maxWindowSize.getValue());
			labelInstanceMask = new int[maxWindowSize.getValue()][numLabels];

			currentK = new int[numLabels];
			for(int i = 0; i < numLabels; i++)
				currentK[i] = 3;

			KmetricHistory = new ArrayList[4][numLabels]; // 1, 3, 5, 7 per label
			for(int i = 0; i < 4; i++)
				for(int j = 0; j < numLabels; j++)
					KmetricHistory[i][j] = new ArrayList<Integer>();


			this.classifierRandom = new Random(this.randomSeedOption.getValue());

			this.contatorInstancias = 0;

			this.instanceMeanIR = new HashMap<Instance, Double>();
			this.globalLabelIR = new double[2][this.numLabels];
			this.globalMeanIR = -1;
			this.globalMaxIR = -1;

		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	//Metodo para treino com as instâncias na janela
	@Override
	public void trainOnInstanceImpl(MultiLabelInstance inst) {
		//Inclui a instancia na janela
		window.add(inst);
		this.contatorInstancias++;
		this.nInstancia++;
//		Atualiza o menor e maior valor de atributos
		updateRanges(inst);

		//Habilita todos os rotulos da nova instancia
		for(int l = 0; l < numLabels; l++)
			labelInstanceMask[window.size()-1][l] = 1;

		int windowSize = window.size();

		//Armazena as distancias dos vizinhos na janela
		get1ToNDistances(inst, window, distanceMatrix[windowSize-1]);

		//ANTES DO TREINO
		this.calaculateWindowMeanIR();

		this.instanceMeanIR.put(inst, 1.0);
		if(this.punishmentType.getChosenLabel().equals("G-PUNISHMENT")){
			updateGlobalIR(inst);
		}
		if(this.punishmentType.getChosenLabel().equals("Dynamic")){
			this.calculateInstanceMeanIR();
		}

		dadosTreino += this.metric.getChosenLabel() + ",";
		dadosTreino += nInstancia + ",";
		dadosTreino += this.window.size() + ",";
		dadosTreino += this.windowMeanIR + ",";
		dadosTreino += this.windowMaxCount + ",";
		dadosTreino += this.windowMaxIR + ",";

		if(this.balancingPosition.getChosenLabel().equals("FIRST")){
    			if (this.window.size() >= 5 && this.window.size() < this.maxWindowSize.getValue() && ((double) contatorInstancias)%((double)this.balancingFrequency.getValue()) == 0){
					if(this.balancingMethod.getChosenLabel().equals("MLSMOTE"))
						MLSMOTE();
					else if(this.balancingMethod.getChosenLabel().equals("MLROS"))
						MLROS();
					else if(this.balancingMethod.getChosenLabel().equals("MLRUS"))
						MLRUS();
			}
		}

		if(!this.punishmentType.getChosenLabel().equals("NO-PUNISHMENT")){
			double threshold = 0;
			//Lista de instancias descartadas
			List<Instance> discarded = new ArrayList<Instance>();
			//Mecanismo de punição
			//Para cada erro cometido por uma instancia na janela
			for(Map.Entry<Instance, Integer> entry : errors.entrySet())
			{
				if(!this.punishmentType.getChosenLabel().equals("Static")){
					threshold = penalty.getValue() * numLabels*this.instanceMeanIR.get(entry.getKey());
				} else {
					threshold = penalty.getValue() * numLabels;
				}


				//Se o valor do erro cometido por aquela instancia ultrapassar o threshold
				if(entry.getValue() > threshold)
				{
					for(int idx = windowSize-1; idx >= 0; idx--)
					{
						//Se aquela for a instancia com o erro cometido
						if(window.get(idx) == entry.getKey())
						{
							//Para cada instancia na janela
							for (int i = idx; i < windowSize-1; i++)
								//Para cada outra instancia na janela
								for (int j = idx; j < i; j++)
									//Copia a distancia do proximo (preparando para remover a instancia da janela
									distanceMatrix[i][j] = distanceMatrix[i+1][j+1];
							//Remove a instancia descartada da mascara de rotulos habilitados por instancia
							for (int i = idx; i < windowSize-1; i++)
								labelInstanceMask[i] = labelInstanceMask[i+1];

							//Adiciona a instancia descartada na lista de descartados
							discarded.add(window.get(idx));

							//Remove a instancia da janela
							window.remove(idx);
							this.instanceMeanIR.remove(idx);

							//Reduz o tamanho da janela
							windowSize--;
							break;
						}
					}
				}
			}

			//remove os erros das instancias descartadas do array de erros
			for(Instance instance : discarded)
				errors.remove(instance);
		}

		//DEPOIS DO MECANISMO DE PUNICAO
		this.calaculateWindowMeanIR();
		dadosTreino += this.window.size() + ",";
		dadosTreino += this.windowMeanIR + ",";
		dadosTreino += this.windowMaxCount + ",";
		dadosTreino += this.windowMaxIR + ",";

		if(this.balancingPosition.getChosenLabel().equals("SECOND")){
			if (this.window.size() >= 5 && this.window.size() < this.maxWindowSize.getValue() && ((double) contatorInstancias)%((double)this.balancingFrequency.getValue()) == 0){
				if(this.balancingMethod.getChosenLabel().equals("MLSMOTE"))
					MLSMOTE();
				else if(this.balancingMethod.getChosenLabel().equals("MLROS"))
					MLROS();
				else if(this.balancingMethod.getChosenLabel().equals("MLRUS"))
					MLRUS();
			}
		}
		//Avalia as subjanelas
		int newWindowSize = getNewWindowSize();

		//Ajusta o tamanho da janela
		if (newWindowSize < windowSize) {
			int diff = windowSize - newWindowSize;
			//Remove os erros das instancias a serem removidas
			for (int i = 0; i < diff; i++)
				errors.remove(window.get(i));
			//Altera a janela em si
			window = window.subList(diff, windowSize);

			//Laco para retirar as distancias armazenadas para as instancias retiradas
			for (int i = 0; i < newWindowSize; i++)
				for (int j = 0; j < i; j++)
					distanceMatrix[i][j] = distanceMatrix[diff+i][diff+j];


			//Retira a mascara de rotulos habilitados por instancia
			for (int i = 0; i < newWindowSize; i++)
				labelInstanceMask[i] = labelInstanceMask[diff+i];
		}

		if (newWindowSize == maxWindowSize.getValue()) {
			//Remove a distancia calculada para o registro mais antigo
			for (int i = 0; i < newWindowSize-1; i++)
				for (int j = 0; j < i; j++)
					distanceMatrix[i][j] = distanceMatrix[i+1][j+1];

			//Remove a instancia da mascara de rotulos habilitados por instancia
			for (int i = 0; i < newWindowSize-1; i++)
				labelInstanceMask[i] = labelInstanceMask[i+1];

			//Remove o erro do registro mais antigo
			errors.remove(window.get(0));

			//Remove a instancia mais antiga
			window.remove(0);
			this.instanceMeanIR.remove(0);

		}

		this.calaculateWindowMeanIR();
		dadosTreino += this.window.size() + ",";
		dadosTreino += this.windowMeanIR + ",";
		dadosTreino += this.windowMaxCount + ",";
		dadosTreino += this.windowMaxIR + "\n";

		if(this.balancingPosition.getChosenLabel().equals("THIRD")){
			if (this.window.size() >= 5 && this.window.size() < this.maxWindowSize.getValue() && ((double) contatorInstancias)%((double)this.balancingFrequency.getValue()) == 0){
				if(this.balancingMethod.getChosenLabel().equals("MLSMOTE"))
					MLSMOTE();
				else if(this.balancingMethod.getChosenLabel().equals("MLROS"))
					MLROS();
				else if(this.balancingMethod.getChosenLabel().equals("MLRUS"))
					MLRUS();
			}
		}
	}

	//Metodo para teste com a instância recebida
	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance instance) {
		//Classe que guarda as predicoes
		MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

		//Vetor que armazena a distancia entre a nova instancia e todas as instancias na janela
		double[] distances = new double[window.size()];

		//Para cada instancia na janela
		for (int i = 0; i < window.size(); i++)

			//Calcula a distancia entre a nova instancia e todas as instancias na janela
			distances[i] = getDistance(instance, window.get(i));

		//Para cada rotulo
		for(int j = 0; j < numLabels; j++)
		{
			//Contador de votos positivos
			int positives = 0;

			//Contador do total de votos
			int totalVotes = 0;

			//Copia do vetor de distancias
			double[] distancesLocal = distances.clone();

			//Flag indicando se o vetor foi modificado
			boolean modify = true;

			//Vetor de booleanos pra marcar quais rotulos contribuiram para o acerto
			boolean[] added = new boolean[KmetricHistory.length];

			//Para cada instancia na janela
			for(int n = 0; n < distancesLocal.length; n++) {

				//Armazena a menor distancia
				double minDistance = Double.MAX_VALUE;

				//Armazena o vizinho mais proximo
				int closestNeighbor = 0;

				//Para cada vizinho mais proximo
				for(int nn = 0; nn < distancesLocal.length; nn++) {
					//Se a distancia for a menor
					if(distancesLocal[nn] <= minDistance) {
						//Armazena a distancia
						minDistance = distancesLocal[nn];
						//Armazena o vizinho mais proximo
						closestNeighbor = nn;
					}
				}

				//
				distancesLocal[closestNeighbor] = Double.MAX_VALUE;

				boolean enter = false;

				//Se o rotulo estiver habilitado
				if(labelInstanceMask[closestNeighbor][j] == 1) {

					//Contabiliza o voto de predicao
					if(window.get(closestNeighbor).classValue(j) == 1)
						positives++;

					totalVotes++;
					enter = true;
					//Linha 9
					// If prediction was misleading, then disable the labelinstance
					if(modify && window.get(closestNeighbor).classValue(j) != instance.classValue(j)) {
						labelInstanceMask[closestNeighbor][j] = 0;

						Integer instanceErrors = errors.remove(window.get(closestNeighbor));

						if(instanceErrors == null)
							errors.put(window.get(closestNeighbor), new Integer(1));
						else
							errors.put(window.get(closestNeighbor), instanceErrors.intValue() + 1);
					}
				} else {
					//Linha 21
					// If label instance was disabled but it would had been a good prediction then reenable
					if(modify && window.get(closestNeighbor).classValue(j) == instance.classValue(j)) {
						labelInstanceMask[closestNeighbor][j] = 1;
					}
				}
				//Nao faria mais sentido poder configurar isso? Quao complicado eh fazer?
				if((totalVotes == 1 || totalVotes == 3 || totalVotes == 5 || totalVotes == 7) && enter == true) {
					double relativeFrequency = positives / (double) totalVotes;
					int labelPrediction = relativeFrequency >= 0.5 ? 1 : 0;
					int kIndex = -1;

					//Poderia ser:
					//kIndex = (totalVotes-1)/2
					if(totalVotes == 1) {
						kIndex = 0;
					}
					else if(totalVotes == 3) {
						kIndex = 1;
					}
					else if(totalVotes == 5) {
						kIndex = 2;
					}
					else if(totalVotes == 7) {
						kIndex = 3;
					}

					if(labelPrediction == instance.classValue(j))
						KmetricHistory[kIndex][j].add(1);
					else
						KmetricHistory[kIndex][j].add(0);

					added[kIndex] = true;

					if(KmetricHistory[kIndex][j].size() > kHistorySize.getValue())
						KmetricHistory[kIndex][j].remove(0);

					if(totalVotes == currentK[j]) {
						prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
						modify = false;
					}

					if(totalVotes == 7) {
						break;
					}
				}
			}

			for(int kIndex = 0; kIndex < KmetricHistory.length; kIndex++) {
				if(added[kIndex] == false) {
					int labelPrediction = 0;

					if(labelPrediction == instance.classValue(j))
						KmetricHistory[kIndex][j].add(1);
					else
						KmetricHistory[kIndex][j].add(0);

					if(KmetricHistory[kIndex][j].size() > kHistorySize.getValue())
						KmetricHistory[kIndex][j].remove(0);
				}
			}
		}

		// Adapt current K to best accurate
		for(int j = 0; j < numLabels; j++) {

			double[] accuracy = new double[KmetricHistory.length];

			for(int kIndex = 0; kIndex < KmetricHistory.length; kIndex++) {

				int sum = 0;

				for(int v = 0; v < KmetricHistory[kIndex][j].size(); v++) {
					sum += KmetricHistory[kIndex][j].get(v);
				}

				accuracy[kIndex] = sum / (double) KmetricHistory[kIndex][j].size();
			}

			int bestAccuracyIndex = -1;
			double bestAccuracyValue = -1;

			for(int kIndex = 0; kIndex < KmetricHistory.length; kIndex++) {
				if(accuracy[kIndex] > bestAccuracyValue) {
					bestAccuracyValue = accuracy[kIndex];
					bestAccuracyIndex = kIndex;
				}
			}

			if(bestAccuracyIndex == 0) {
				currentK[j] = 1;
			}
			if(bestAccuracyIndex == 1) {
				currentK[j] = 3;
			}
			if(bestAccuracyIndex == 2) {
				currentK[j] = 5;
			}
			if(bestAccuracyIndex == 3) {
				currentK[j] = 7;
			}
		}

		return prediction;
	}

	public void MLROS(){

		Random random = this.classifierRandom;
		MultiLabelInstance synt;
		int samplesToClone = (int) Math.ceil(this.window.size()*1.0/100*this.minorityIncrement.getValue());
		//Calcula a taxa de desbalanceamento medio
		calaculateWindowMeanIR();
		double irlbl;

		int newInstances = 0, totalInstances = 0;

		//Lista que armazena todas as instancias de um rotulo
		List<Instance> minBag;

		int aux;
		//Para cada rotulo acima da taxa de desbalanceamento media, cria um conjunto de instancias sinteticas para balancear o rotulo
		for (int l = 0; l < numLabels; l++){

			if (window.size() >= maxWindowSize.getValue()-1) {
				return;
			}
			newInstances = 0;

			//Armazena a taxa de desbalanceamento de um rotulo especifico
			irlbl = this.labelIR[l];

			//Se o desbalanceamento do rotulo atual for maior que o desbalanceamento medio
			if(irlbl > this.windowMeanIR){

				//Traz todas as instancias do rotulo selecionado
				minBag = getAllInstancesOfLabel(l);
				if(minBag == null) continue;

				aux = 0;
				//Para cada instancia que contem o rotulo selecionado
				while(aux < samplesToClone && irlbl > this.windowMeanIR) {

					synt = (MultiLabelInstance) minBag.get(random.nextInt(minBag.size())).copy();
					newInstances++;

					//Incluí a instancia na janela (no prequential, esse passo acontece depois do evaluation
					window.add(synt);

					//CORRIGIR PARA COPIAR AS HABILITACOES DO ROTULO COPIADO
					//Habilita todos os rotulos da nova instancia
					for(int l2 = 0; l2 < numLabels; l2++)
						labelInstanceMask[window.size()-1][l2] = 1;

					if (window.size() >= maxWindowSize.getValue()-2) {
						return;
					}

					irlbl = calculateIRLBL(l);
					aux++;
				}
			}
			totalInstances += newInstances;
		}
	}

	public void MLRUS(){

		Random random = this.classifierRandom;

		//Calcula a taxa de desbalanceamento medio
		double irlbl;

		int deletedInstance;
		int totalInstances = 0;

		//Lista que armazena todas as instancias de um rotulo
		List<Integer> minBag;

		int aux, removed;
		int windowSize = this.window.size();
		List<Instance> discarded = new ArrayList<Instance>();
		//Para cada rotulo acima da taxa de desbalanceamento media, cria um conjunto de instancias sinteticas para balancear o rotulo
		for (int l = 0; l < numLabels; l++){

			deletedInstance = 0;

			//Armazena a taxa de desbalanceamento de um rotulo especifico
			irlbl = this.labelIR[l];

			//Se o desbalanceamento do rotulo atual for maior que o desbalanceamento medio
			if(irlbl <= this.windowMeanIR){

				//Traz todas as instancias do rotulo selecionado
				minBag = getAllIndexOfLabel(l);
				if(minBag == null) continue;

				aux = 0;
				//Para cada instancia que contem o rotulo selecionado
				while(minBag.size() <= minBag.size()*(1-this.majorityReduction.getValue()) && minBag != null && minBag.size() > 0) {
					removed = random.nextInt(minBag.size());

					//Remove a instancia descartada da mascara de rotulos habilitados por instancia
					for (int i = minBag.get(removed); i < windowSize-1; i++)
						labelInstanceMask[i] = labelInstanceMask[i+1];

					//Adiciona a instancia descartada na lista de descartados
					discarded.add(window.get(minBag.get(removed)));

					//Remove a instancia da janela
					window.remove(minBag.get(removed));
					this.instanceMeanIR.remove(minBag.get(removed));

					//Reduz o tamanho da janela
					windowSize--;
					minBag.remove(removed);
					deletedInstance++;
				}
			}


			//remove os erros das instancias descartadas do array de erros
			for(Instance instance : discarded)
				errors.remove(instance);
			totalInstances += deletedInstance;

		}
	}

	public void MLSMOTE(){
		MultiLabelInstance synt;

		//Linha 2 do pseudocodigo
		//Calcula a taxa de desbalanceamento medio
//		calculateMeanIR();
		//Variavel que armazenara os votos dos vizinhos
		int votes;

		//Vetor que armazena as distancias entre um rotulo e os demais
		double[] distancesVector;

		//Lista que armazena todas as instancias de um rotulo
		List<Instance> minBag;

		//Variavel auxiliar que armazena a menor distancia
		double minDistance;

		//Variavel auxiliar que armazena o id da instancia mais proxima
		int closestInstance;

		//Lista que armazena as melhores instancias
		List<Instance> bestInstances = new ArrayList<Instance>();
		Instance sample;
		int newInstances;
		int totalInstances = 0;

		//Laco da linha 3
		//Para cada rotulo acima da taxa de desbalanceamento media, cria um conjunto de instancias sinteticas para balancear o rotulo
		for (int l = 0; l < numLabels; l++){

			if (window.size() >= maxWindowSize.getValue()-1) {
				return;
			}

			newInstances = 0;
			//Linha 5
			//Se o desbalanceamento do rotulo atual eh maior que o desbalanceamento medio
			if(this.labelIR[l] > this.windowMeanIR){
				//linha 7
				//Traz todas as instancias do rotulo selecionado
				minBag = getAllInstancesOfLabel(l);

				//Variavel auxiliar para armazenar os votos
				votes = 0;

				//Linha 8
				if (minBag == null || minBag.size() <= 1) {
					continue;
				} else {


					//Inicializa o vetor auxiliar de distancias
					distancesVector = new double[minBag.size()-1];
					//Para cada instancia que contem o rotulo selecionado
					for(int m = 0; m < minBag.size(); m++) {
						sample = (MultiLabelInstance) minBag.remove(0);

						//Linha 9
						//Calcula a distancia entre a instancia atual e as outras instancias que contem o rotulo selecionado
						get1ToNDistances(sample, minBag, distancesVector);

						bestInstances = new ArrayList<Instance>();
						int votacao_maxima = 0;
						if(mlsmoteK.getValue() <= minBag.size()) votacao_maxima = mlsmoteK.getValue();
						else votacao_maxima = minBag.size();

						//Laco para selecionar os k vizinhos mais proximos
						for (int i = 0; i < votacao_maxima; i++) {
							minDistance = Double.MAX_VALUE;
							closestInstance = 0;
							for (int j = 0; j < distancesVector.length; j++) {
								if (distancesVector[j] < minDistance) {
									closestInstance = j;
								}
							}
							distancesVector[closestInstance] = Double.MAX_VALUE;
							bestInstances.add(minBag.get(closestInstance));
						}

						int rank = mlsmoteK.getValue();
						if(mlsmoteK.getValue() > minBag.size())  rank = minBag.size();

						synt = (MultiLabelInstance) newSample(sample, bestInstances.get(this.classifierRandom.nextInt(rank)), bestInstances);
						newInstances++;

						//Inclui a nova instancia na janela
						window.add(synt);

						this.instanceMeanIR.put(synt, 1.0);

						//Atualiza o menor e maior valor de atributos
						updateRanges(synt);

						//Habilita todos os rotulos da nova instancia
						for(int l2 = 0; l2 < numLabels; l2++)
							labelInstanceMask[window.size()-1][l2] = 1;

						//Armazena as distancias dos vizinhos na janela
						get1ToNDistances(synt, window, distanceMatrix[window.size()-1]);

						if (window.size() >= maxWindowSize.getValue()-1) {
							totalInstances += newInstances;
							return;
						}
						minBag.add(sample);

					}
				}
				totalInstances += newInstances;
			}
		}
	}


	public Instance newSample(Instance sample, Instance refNeigh, List<Instance> bestInstances) {
		double newNumeric;
		double[] newLabels;

		//Cria uma copia da instancia para editar
		Instance s = sample.copy();

		ArrayList<Double> attributeValues = new ArrayList<Double>();
		ArrayList<Integer> attributeIndexes = new ArrayList<Integer>();
		newNumeric = 0;
		//Cria instancia nova
		if (instanceType.getChosenLabel().equals("SPARSE")) {
			//Cria rotulos da instancia nova
			newLabels = new double[numLabels];
			//INTERSECAO
			if (labelGenerationMethod.getChosenIndex() == 0) {
				//Calcula a frequencia dos rotulos
				for (int t = 0; t < numLabels; t++) {
					newLabels[t] = sample.classValue(t);
					for (Instance l : bestInstances) {
						newLabels[t] += l.classValue(t);
					}
				}

				//Para cada rotulo, a presenca daquele rotulo na nova instancia eh determinada nesse laco
				for (int i = 0; i < numLabels; i++) {
					//O rotulo precisa estar presente em todas as instancias selecionadas, ou seja, se a frequencia for igual ao tamanho da lista
					if (newLabels[i] == bestInstances.size() + 1) {
						attributeIndexes.add(i);
						attributeValues.add(1.0);
					}
				}

				//UNIAO
			} else if (labelGenerationMethod.getChosenIndex() == 1) {
				//Calcula a frequencia dos rotulos
				for (int i = 0; i < numLabels; i++) {
					newLabels[i] = sample.classValue(i);
					for (Instance l : bestInstances) {
						newLabels[i] += l.classValue(i);
					}
				}

				//Para cada rotulo, a presenca daquele rotulo na nova instancia eh determinada nesse laco
				for (int i = 0; i < numLabels; i++) {

					//O rotulo precisa estar presente pelo menos uma das instancias, ou seja, sua frequencia eh maior que zero
					if (newLabels[i] > 0) {
						attributeIndexes.add(i);
						attributeValues.add(1.0);
					}
				}

				//RANKING
			} else if (labelGenerationMethod.getChosenIndex() == 2) {
				//Calcula a frequencia dos rotulos
				for (int i = 0; i < numLabels; i++) {
					newLabels[i] = sample.classValue(i);
					for (Instance l : bestInstances) {
						newLabels[i] += l.classValue(i);
					}
				}

				//Para cada rotulo, a presenca daquele rotulo na nova instancia eh determinada nesse laco
				for (int i = 0; i < numLabels; i++) {
					//Se o rotulo estiver presente na maioria, ou seja, a frequencia eh maior que o chao da divisao por dois do (k + 1)
					if (newLabels[i] > (int) ((mlsmoteK.getValue() + 1) / 2)) {
						attributeIndexes.add(i);
						attributeValues.add(1.0);
					}
				}
			} else {
				System.out.println("ERROR IN LABEL GENERATION");
			}

			for (int i = sample.numOutputAttributes(); i < sample.numAttributes(); i++) {
				if (isNumeric("" + sample.value(i))) {
					newNumeric = Math.abs(sample.value(i) - refNeigh.value(i));
					newNumeric = newNumeric * this.classifierRandom.nextDouble();
					newNumeric = Math.round(sample.value(i) + newNumeric);

					if (newNumeric > 0) {
						attributeIndexes.add(i);
						attributeValues.add(1.0);
					}
				} else {
					System.out.println("Atributo não numérico");
				}
			}
			double[] auxValores = new double[attributeIndexes.size()];
			int[] auxindices = new int[attributeIndexes.size()];

			for(int i = 0; i < attributeIndexes.size(); i++){
				auxValores[i] = attributeValues.get(i);
				auxindices[i] = attributeIndexes.get(i);
			}

			s.addSparseValues(auxindices, auxValores, auxindices.length);

		} else if (instanceType.getChosenLabel().equals("DENSE")) {
			s = sample.copy();
			for (int i = s.numOutputAttributes(); i < s.numAttributes(); i++) {
				newNumeric = Math.abs(s.value(i) - refNeigh.value(i));
				newNumeric = newNumeric * this.classifierRandom.nextDouble();
				newNumeric = Math.round(s.value(i) + newNumeric);
				s.setValue(i, newNumeric);
			}

			//Cria rotulos da instancia nova
			newLabels = new double[numLabels];

			//INTERSECAO
			if (labelGenerationMethod.getChosenIndex() == 0) {
				//Calcula a frequencia dos rotulos
				for (int t = 0; t < numLabels; t++) {
					newLabels[t] = sample.classValue(t);
					for (Instance l : bestInstances) {
						newLabels[t] += l.classValue(t);
					}
				}

				//Para cada rotulo, a presenca daquele rotulo na nova instancia eh determinada nesse laco
				for (int i = 0; i < numLabels; i++) {
					//O rotulo precisa estar presente em todas as instancias selecionadas, ou seja, se a frequencia for igual ao tamanho da lista
					if (newLabels[i] == bestInstances.size() + 1) {
						s.setValue(i, 1);
					} else s.setValue(i, 0);
				}

				//UNIAO
			} else if (labelGenerationMethod.getChosenIndex() == 1) {
				//Calcula a frequencia dos rotulos
				for (int i = 0; i < numLabels; i++) {
					newLabels[i] = sample.classValue(i);
					for (Instance l : bestInstances) {
						newLabels[i] += l.classValue(i);
					}
				}

				//Para cada rotulo, a presenca daquele rotulo na nova instancia eh determinada nesse laco
				for (int i = 0; i < numLabels; i++) {
					//O rotulo precisa estar presente pelo menos uma das instancias, ou seja, sua frequencia eh maior que zero
					if (newLabels[i] > 0) {
						s.setValue(i, 1);
					} else s.setValue(i, 0);
				}

				//RANKING
			} else if (labelGenerationMethod.getChosenIndex() == 2) {
				//Calcula a frequencia dos rotulos
				for (int i = 0; i < numLabels; i++) {
					newLabels[i] = sample.classValue(i);
					for (Instance l : bestInstances) {
						newLabels[i] += l.classValue(i);
					}
				}

				//Para cada rotulo, a presenca daquele rotulo na nova instancia eh determinada nesse laco
				for (int i = 0; i < numLabels; i++) {
					//Se o rotulo estiver presente na maioria, ou seja, a frequencia eh maior que o chao da divisao por dois do (k + 1)
					if (newLabels[i] > (int) ((mlsmoteK.getValue() + 1) / 2)) {
						s.setValue(i, 1);
					} else s.setValue(i, 0);
				}
			} else {
				s = sample.copy();
				System.out.println("ERROR IN LABEL GENERATION");
			}
		}

		return s;
	}

	public static boolean isNumeric(String strNum) {
		if (strNum == null) {
			return false;
		}
		try {
			double d = Double.parseDouble(strNum);
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}


	public String mostFrequentVal(int posicao, List<Instance> bestInstances){

		return "";
	}
	public void updateGlobalIR(Instance instance){


		for (int l = 0; l < numLabels; l++){
			this.globalLabelIR[0][l] += instance.classValue(l);
			if (this.globalLabelIR[0][l] > this.globalMaxCount) this.globalMaxCount = this.globalLabelIR[0][l];

		}


		//Variavel que armazena a quantidade de rotulos que estao presentes em pelo menos uma instancia
		int rotulosPresentes = 0;

		//Rotulo para calcular a taxa de desbalanceamento media
		for (int l = 0; l < numLabels; l++){

			if(this.globalLabelIR[0][l] > 0){

				this.globalLabelIR[1][l] = this.globalMaxCount*1.0/this.globalLabelIR[0][l];

				if (this.globalLabelIR[1][l] > this.globalMaxIR) this.globalMaxIR = this.globalLabelIR[1][l];
				this.globalMeanIR += this.globalLabelIR[0][l];
				rotulosPresentes++;
			}
		}



		this.globalMeanIR = this.globalMeanIR/rotulosPresentes;
		double auxMeanIR = 0;
		for (Instance w: this.window) {
			auxMeanIR = 0;
			for (int l = 0; l < numLabels; l++){
				if (w.classValue(l) == 1) {
					auxMeanIR += this.globalLabelIR[0][l];
				}
			}
			auxMeanIR = auxMeanIR/rotulosPresentes;
			this.instanceMeanIR.replace(w, auxMeanIR);
		}

		globalMaxCount = 0;
	}

	//Metodo para calcular a taxa de desbalanceamento media da janela
	public void calaculateWindowMeanIR(){

		this.windowMaxIR = -1;
		this.windowMeanIR = 0;
		//Variavel que armazena a quantidade de rotulos que estao presentes em pelo menos uma instancia
		this.windowPresentLabels = 0;
		this.labelIR = new double[this.numLabels];


		//Laco para contar a presenca de cada rotulo nas instancias da janela e identificar o rotulo mais presente
		for (int l = 0; l < numLabels; l++){
			for (Instance w: this.window) {
				this.labelIR[l] += w.classValue(l);
			}
			if (this.labelIR[l] > this.windowMaxCount) this.windowMaxCount = this.labelIR[l];
		}
		//Laco para calcular a taxa de desbalanceamento media
		for (int l = 0; l < numLabels; l++){
			//Para garantir que nao ha
			if(this.labelIR[l] > 0){
				this.labelIR[l] = this.windowMaxCount*1.0/this.labelIR[l];
				if (this.labelIR[l] > this.windowMaxIR) this.windowMaxIR = this.labelIR[l];
				this.windowMeanIR += this.labelIR[l];
				windowPresentLabels++;
			}
		}
		this.windowMeanIR = this.windowMeanIR/windowPresentLabels;
	}


	public void calculateInstanceMeanIR(){
		double instanceMeanIR;
		int instancePresentLabels;
		instancePresentLabels = 0;
		for (Instance w: this.window) {
			instanceMeanIR = 0;
			for (int l = 0; l < numLabels; l++){
				if (w.classValue(l) == 1) {
					instanceMeanIR += this.labelIR[l];
					instancePresentLabels++;
				}
			}
			instanceMeanIR = instanceMeanIR/instancePresentLabels;
			this.instanceMeanIR.replace(w, instanceMeanIR);
		}

	}



	//Metodo para calcular a taxa de desbalanceamento de um rótulo específico
	public double calculateIRLBL(int l){

		int presenca = 0;
		double irlbl = 0;

		for (Instance w: this.window) {
			presenca += w.classValue(l);
		}

		if(presenca > this.windowMaxIR){
			this.windowMeanIR = presenca;
		}
		irlbl = this.windowMaxIR*1.0/presenca;

		return irlbl;
	}


	public List<Instance> getAllInstancesOfLabel(int l){
		List<Instance> listOfInstances = new ArrayList<Instance>();

		for (Instance w: this.window){
			if(w.classValue(l) == 1){
				listOfInstances.add((MultiLabelInstance) w);
			}
		}

		return listOfInstances;
	}


	public List<Integer> getAllIndexOfLabel(int l){
		List<Integer> listOfInstances = new ArrayList<Integer>();
		int aux = 0;
		for (Instance w: this.window){
			if(w.classValue(l) == 1){
				listOfInstances.add(aux);
			}
			aux++;
		}

		return listOfInstances;
	}


	private Double getMetricSums(Instance instance, MultiLabelPrediction prediction) {
		double correct = 0;
		int tp, fp,fn, tn;

		tp = fp = tn = fn = 0;
		double[] predicoes = new double[prediction.numOutputAttributes() + 5];
		/** preset threshold */
		double t = 0.5;

		for (int j = 0; j < prediction.numOutputAttributes(); j++) {
			int yp = (prediction.getVote(j, 1) >= t) ? 1 : 0;

			if(this.metric.getChosenLabel().equals("F-Score") || this.metric.getChosenLabel().equals("G-Mean")){

				if(yp == 1 && (int) instance.classValue(j) == 1) tp += 1;
				else if(yp == 1 && (int) instance.classValue(j) == 0) fp += 1;
				else if(yp == 0 && (int) instance.classValue(j) == 0) tn += 1;
				else if(yp == 0 && (int) instance.classValue(j) == 1) fn += 1;

			} else {
				correct += ((int) instance.classValue(j) == yp) ? 1 : 0;
			}

		}
		if(this.metric.getChosenLabel().equals("F-Score")){
			if((2*tp + fp + fn) == 0) {
				correct = 0;
			}else{
				correct = 2*tp/(2*tp + fp + fn);
			}
		}else if(this.metric.getChosenLabel().equals("G-Mean")){
			int sensitivity, specificity;

			if(tp + fn == 0 || tn + fp == 0){
				correct = 0;
			}else{
				sensitivity = tp/(tp + fn);
				specificity = tn/(tn + fp);
				correct = sensitivity*specificity;
			}
		}

		return correct;
	}

	private double getMetricFromHistory(List<Double> history) {

		double metric = 0.0;

		if(this.metric.getChosenLabel().equals("Subset Accuracy"))
		{
			for(Double instanceSum : history)
				metric += (instanceSum == numLabels) ? 1 : 0;
		}
		else if (this.metric.getChosenLabel().equals("Hamming Score"))
		{
			for(Double instanceSum : history)
				metric += instanceSum / (double) numLabels;
		}
		else if (this.metric.getChosenLabel().equals("F-Score"))
		{
			for(Double instanceSum : history)
				metric += instanceSum;
		}
		else if (this.metric.getChosenLabel().equals("G-Mean"))
		{
			for(Double instanceSum : history)
				metric += instanceSum;

			return Math.sqrt(metric) / history.size();
		}

		return metric / history.size();
	}

	/**
	 * Computes the Euclidean distance between one sample and a collection of samples in an 1D-array.
	 */
	private void get1ToNDistances(Instance sample, List<Instance> samples, double[] distances) {

		for (int i = 0; i < samples.size(); i++)
			distances[i] = getDistance(sample, samples.get(i));
	}

	/**
	 * Returns the Euclidean distance.
	 */
	private double getDistance(Instance instance1, Instance instance2) {

		String d = distance.getChosenLabel();

		double distance = 0;

		if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
		{
			for(int i = 0; i < instance1.numInputAttributes(); i++)
			{
				double val1 = instance1.valueInputAttribute(i);
				double val2 = instance2.valueInputAttribute(i);

				if(attributeRangeMax[i] - attributeRangeMin[i] != 0) {

					val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					if(d.equals("MANHATTAN")) distance += Math.abs((val1 - val2));
					else distance += (val1 - val2) * (val1 - val2);
				}

			}
		}
		else // Sparse Instance
		{
			int firstI = -1, secondI = -1;
			int firstNumValues  = instance1.numValues();
			int secondNumValues = instance2.numValues();
			int numAttributes   = instance1.numAttributes();
			int numOutputs      = instance1.numOutputAttributes();

			for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

				if (p1 >= firstNumValues) {
					firstI = numAttributes;
				} else {
					firstI = instance1.index(p1);
				}

				if (p2 >= secondNumValues) {
					secondI = numAttributes;
				} else {
					secondI = instance2.index(p2);
				}

				if (firstI < numOutputs) {
					p1++;
					continue;
				}

				if (secondI < numOutputs) {
					p2++;
					continue;
				}

				if (firstI == secondI) {
					int idx = firstI - numOutputs;
					if (attributeRangeMax[idx] - attributeRangeMin[idx] != 0) {
						double val1 = instance1.valueSparse(p1);
						double val2 = instance2.valueSparse(p2);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						if(d.equals("MANHATTAN")) distance += Math.abs((val1 - val2));
						else distance += (val1 - val2) * (val1 - val2);

					}
					p1++;
					p2++;
				} else if (firstI > secondI) {
					int idx = secondI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val2 = instance2.valueSparse(p2);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						//CALCULO DA DISTANCIA MANHATTAN
						if(d.equals("MANHATTAN")) distance += Math.abs((val2));
						else distance += (val2) * (val2);
					}
					p2++;
				} else {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						if(d.equals("MANHATTAN")) distance += Math.abs((val1));
						else distance += (val1) * (val1);
					}
					p1++;
				}
			}

		}
		if(d.equals("MANHATTAN")) return distance;
		else return Math.sqrt(distance);
	}

	private void updateRanges(MultiLabelInstance instance) {
		for(int i = 0; i < instance.numInputAttributes(); i++)
		{
			if(instance.valueInputAttribute(i) < attributeRangeMin[i])
				attributeRangeMin[i] = instance.valueInputAttribute(i);
			if(instance.valueInputAttribute(i) > attributeRangeMax[i])
				attributeRangeMax[i] = instance.valueInputAttribute(i);
		}
	}
	//Metodo que define o novo tamanho da janela
	/**
	 * Returns the bisected size which maximized the metric
	 */
	private int getNewWindowSize() {

		int numSamples = window.size();

		//Se o tamanho da janela for menor que o dobro do minimo, nao sera feita nenhuma alteracao no tamanho.
		//Isso eh necessario, pois a janela eh subdividida pela metade para a avaliacao
		if (numSamples < 2 * minWindowSize.getValue())
			return numSamples;
		else {
			List<Integer> numSamplesRange = new ArrayList<Integer>();
			numSamplesRange.add(numSamples);
			while (numSamplesRange.get(numSamplesRange.size() - 1) >= 2 * minWindowSize.getValue())
				numSamplesRange.add((int) (numSamplesRange.get(numSamplesRange.size() - 1) * reductionRatio.getValue()));

			//Estrutura auxiliar de iterador para remover os elementos que nao devem ser considerados no historico de previsao para avaliacao da subjanela
			Iterator<Integer> it = predictionHistories.keySet().iterator();
			while (it.hasNext()) {
				Integer key = (Integer) it.next();
				if (!numSamplesRange.contains(numSamples - key))
					it.remove();
			}

			List<Double> metricList = new ArrayList<Double>();
			for (Integer numSamplesIt : numSamplesRange) {
				int idx = numSamples - numSamplesIt;
				List<Double> predHistory;
				if (predictionHistories.containsKey(idx))
					predHistory = getIncrementalTestTrainPredHistory(window, idx, predictionHistories.get(idx));
				else
					predHistory = getTestTrainPredHistory(window, idx);

				predictionHistories.put(idx, predHistory);

				metricList.add(getMetricFromHistory(predHistory));


			}
			int maxMetricIdx = metricList.indexOf(Collections.max(metricList));
			int windowSize = numSamplesRange.get(maxMetricIdx);

			if (windowSize < numSamples)
				adaptHistories(maxMetricIdx);

			return windowSize;
		}
	}

	/**
	 * Returns the n smallest indices of the smallest values (sorted).
	 */
	private int[] nArgMin(int n, double[] values, int startIdx, int endIdx, int label) {

		int indices[] = new int[n];

		for (int i = 0; i < n; i++){
			double minValue = Double.MAX_VALUE;
			for (int j = startIdx; j < endIdx + 1; j++){

				if (labelInstanceMask[j][label] == 1 && values[j] < minValue){
					boolean alreadyUsed = false;
					for (int k = 0; k < i; k++){
						if (indices[k] == j){
							alreadyUsed = true;
						}
					}
					if (!alreadyUsed){
						indices[i] = j;
						minValue = values[j];
					}
				}
			}
		}
		return indices;
	}

	public int[] nArgMin(int n, double[] values, int label) {
		return nArgMin(n, values, 0, values.length-1, label);
	}

	/**
	 * Returns the votes for each label.
	 */
	private double[] getPrediction(int[] nnIndices, List<Instance> instances, int j) {

		int count = 0;

		for (int nnIdx : nnIndices)
			if(instances.get(nnIdx).classValue(j) == 1)
				count++;

		double relativeFrequency = count / (double) nnIndices.length;

		return new double[]{1.0 - relativeFrequency, relativeFrequency};
	}



	/**
	 * Creates a prediction history from the scratch.
	 */
	private List<Double> getTestTrainPredHistory(List<Instance> instances, int startIdx) {

		List<Double> predictionHistory = new ArrayList<Double>();

		for (int i = startIdx; i < instances.size(); i++) {

			MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

			for(int l = 0; l < numLabels; l++) {
				int nnIndices[] = nArgMin(Math.min(currentK[l], i - startIdx), distanceMatrix[i], startIdx, i-1 ,l);
				prediction.setVotes(l, getPrediction(nnIndices, instances, l));
			}

			predictionHistory.add(getMetricSums(instances.get(i), prediction));
		}

		return predictionHistory;
	}

	/**
	 * Creates a prediction history incrementally by using the previous predictions.
	 */
	private List<Double> getIncrementalTestTrainPredHistory(List<Instance> instances, int startIdx, List<Double> predictionHistory) {

		for (int i = startIdx + predictionHistory.size(); i < instances.size(); i++) {
			MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

			for(int l = 0; l < numLabels; l++) {
				int nnIndices[] = nArgMin(Math.min(currentK[l], distanceMatrix[i].length), distanceMatrix[i], startIdx, i-1, l);
				prediction.setVotes(l, getPrediction(nnIndices, instances, l));
			}

			predictionHistory.add(getMetricSums(instances.get(i), prediction));
		}

		return predictionHistory;
	}

	/**
	 * Removes predictions of the largest window size and shifts the remaining ones accordingly.
	 */
	private void adaptHistories(int numberOfDeletions) {
		for (int i = 0; i < numberOfDeletions; i++){
			SortedSet<Integer> keys = new TreeSet<Integer>(predictionHistories.keySet());
			predictionHistories.remove(keys.first());
			keys = new TreeSet<Integer>(predictionHistories.keySet());
			for (Integer key : keys){
				List<Double> predHistory = predictionHistories.remove(key);
				predictionHistories.put(key-keys.first(), predHistory);
			}
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public boolean isRandomizable() {
		return true;
	}

	//Metodo para reiniciar o modelo de classificação
	@Override
	public void resetLearningImpl() {
		if(window != null)
		{
			window.clear();
			this.dadosTreino = "";
			distanceMatrix = new double[maxWindowSize.getValue()][maxWindowSize.getValue()];
			predictionHistories = new HashMap<Integer, List<Double>>();
			errors = new HashMap<Instance, Integer>();
			labelInstanceMask = new int[maxWindowSize.getValue()][numLabels];

			currentK = new int[numLabels];
			for(int i = 0; i < numLabels; i++)
				currentK[i] = 3;

			KmetricHistory = new ArrayList[4][numLabels]; // 1, 3, 5, 7 per label
			for(int i = 0; i < 4; i++)
				for(int j = 0; j < numLabels; j++)
					KmetricHistory[i][j] = new ArrayList<Integer>();

			this.contatorInstancias = 0;
		}
	}

	public void resetDadosTreino(){

		this.dadosTreino = "";
	}


	// ------- following are private debugging functions -----------
	public static void main(String args[]) {
		// test routines
	}

	private void printMatrix(double M[][]) {
		System.out.println("--- MATRIX ---");
		for (int i = 0; i < M.length; i++) {
			for (int j = 0; j < M[i].length; j++) {
				System.out.print(" " + Utils.doubleToString(M[i][j], 5, 3));
			}
			System.out.println("");
		}
	}

	private void printVector(double V[]) {
		System.out.println("--- VECTOR ---");
		for (int j = 0; j < V.length; j++) {
			System.out.print(" " + Utils.doubleToString(V[j], 5, 3));
		}
		System.out.println("");
	}

	public void imprimeAtributos(Instance s){
		System.out.println("ATRIBUTOS");
		for(int i = 0; i < s.numInputAttributes(); i++){
			System.out.print(s.value(i) + ",");
		}
		System.out.println();
		System.out.println("ROTULOS");
		for(int i = 0; i < s.numInputAttributes(); i++){
			System.out.print(s.classValue(i) +",");
		}
		System.out.println();
	}
}