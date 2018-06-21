package AP;

import org.json.JSONArray;

import edu.stanford.nlp.ling.CoreAnnotations;
import java.io.StringReader;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.LexicalizedParserQuery;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.trees.PennTreeReader;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeReader;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ScoredObject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Properties;
import java.util.Queue;

import org.json.JSONException;
import org.json.JSONObject;
import com.google.common.collect.Lists;

class simple {

	private final static String RNN_MODEL = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
	private final TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(),
			"invertible=true");
	private final StanfordCoreNLP pipeline;
	private static final LexicalizedParser parser2 = LexicalizedParser.loadModel(RNN_MODEL);
	public static List<ScoredObject<Tree>> kBest;
	public static List<Double> list = new ArrayList<Double>();
	public static List<Double> list1 = new ArrayList<Double>();
	public static String tri;
	static String sentence = "He lifted the elephant with one hand.";

	public simple() {
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos");
		pipeline = new StanfordCoreNLP(props);
	}

	public Tree parsernn(String str) {
		List<CoreLabel> tokens = tokenize(str);
		Tree tree2 = parser2.apply(tokens);
		return tree2;
	}

	private List<CoreLabel> tokenize(String str) {
		Tokenizer<CoreLabel> tokenizer = tokenizerFactory.getTokenizer(new StringReader(str));
		return tokenizer.tokenize();
	}

	public JSONArray parse(String text, Integer maxDepth) throws JSONException, IOException {
		Annotation document = new Annotation(text);
		pipeline.annotate(document);

		LexicalizedParserQuery lpq = (LexicalizedParserQuery) parser2.parserQuery();
		List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

		TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
		Tokenizer<CoreLabel> tok = tokenizerFactory.getTokenizer(new StringReader(text));

		List<CoreLabel> rawWords2 = tok.tokenize();
		lpq.parse(rawWords2);
		kBest = lpq.getKBestPCFGParses(2);

		for (ScoredObject<Tree> one : kBest) {
			Double samplearray = one.score();
			list.add(samplearray);
			System.out.println(list);
		}
		System.out.println(Arrays.asList(list.toString().trim().split(" ")));

		String[] splitStringArray = list.toString().split("  ");
		System.out.println("Tree probabilities: " + Arrays.asList(splitStringArray));

		Double d = Double.parseDouble(list.get(0).toString());
		d = Math.exp(d) * Math.pow(10, 21);
		// d=(double)Math.round(d * 1000d) / 1000d;
		Double e = Double.parseDouble(list.get(1).toString());
		e = Math.exp(e) * Math.pow(10, 21);
		// e=(double)Math.round(e * 1000d) / 1000d;

		// f=(double)Math.round(f * 1000d) / 1000d;

		Double t = d + e;
		d = d / t;
		e = e / t;

		System.out.println("ROOT 1: " + d);
		System.out.println("ROOT 2: " + e);

		list1.add(d);
		list1.add(e);

		JSONArray array = new JSONArray();
		for (ScoredObject<Tree> each : kBest) {
			System.out.println(each + "\n" + "-------------------------------------------------------------");

			tri = each.toString();
			// tri=tri.replaceAll("\\d","");
			tri = tri.replace(",", "").replace("@", "").replace("]", "").replace("-", "").trim();

			System.out.println(Arrays.asList(tri.trim().split(" ")));

			TreeReader r = new PennTreeReader(new StringReader(tri));
			Tree best = r.readTree();
			System.out.println(best);

			List<CoreLabel> coreLabels = null;
			for (CoreMap sentence : sentences) {
				coreLabels = sentence.get(CoreAnnotations.TokensAnnotation.class);
				System.out.println(sentences.size());
			}
			array.put(toJSON(best, coreLabels.iterator()));
		}

		System.out.println(array);

		if (maxDepth != 0)
			array = refineArray(array, maxDepth); // returns only required part of the array.
		list.clear();
		list1.clear();
		return array;
	}

	public static JSONObject toJSON(Tree best, Iterator<CoreLabel> labels) throws JSONException, IOException {
		List<JSONObject> children = Lists.newArrayList();
		for (Tree child : best.getChildrenAsList()) {
			children.add(toJSON(child, labels));
		}

		JSONObject obj = new JSONObject();

		if (best.isLeaf()) {
			CoreLabel next = labels.next();
			String word = next.get(CoreAnnotations.TextAnnotation.class);
			obj.put("word", word);
			obj.put("type", "TK");
		} else {
			obj.put("type", best.label());
		}

		return new JSONObject().put("data", obj).put("children", new JSONArray(children)).put("Root", list1)
				.put("RRoot", list);
	}

	/****
	 * Returns breadth first search traversal of the tree with a specified depth
	 * 
	 * @param jsonObj
	 * @param maxDepth
	 * @return
	 * @throws JSONException
	 */
	public List<List<String>> breadthFirstSearch(JSONObject jsonObj, int maxDepth) throws JSONException {
		List<List<String>> levels = new ArrayList<List<String>>();

		Queue<List<JSONObject>> queue = new LinkedList<List<JSONObject>>();
		Queue<Integer> count = new LinkedList<Integer>();

		// Add the head JSONObj to the queue within a list.
		List<JSONObject> objList = new ArrayList<>();
		objList.add(jsonObj);
		queue.add(objList);
		count.add(1);
		// We need to grow the list unless the queue is empty
		while (!queue.isEmpty() && levels.size() < maxDepth + 2) {
			List<JSONObject> polledList = queue.poll();
			List<String> level = new ArrayList<String>();
			objList = new ArrayList<JSONObject>();
			for (JSONObject json : polledList) {
				level.add(json.getJSONObject("data").getString("type"));
				if (null != json && json.getJSONArray("children").length() != 0) {
					JSONArray jsonArr = json.getJSONArray("children");
					for (int i = 0; i < jsonArr.length(); i++) {
						JSONObject jsonObjLeft = json.getJSONArray("children").getJSONObject(i);
						objList.add(jsonObjLeft);
					}
				}
			}
			// Don't add to the queue if the list is empty, we want the loop to stop!
			if (!objList.isEmpty())
				queue.add(objList);
			levels.add(level);
		}
		return levels;
	}

	/****
	 * Reduces the array if the trees are identical till the level defined by
	 * MAX_DEPTH Depth is counted by ignoring ROOT and S.
	 * 
	 * @param array
	 * @return
	 * @throws JSONException
	 */
	public JSONArray refineArray(JSONArray array, Integer maxDepth) throws JSONException {
		boolean reduce = true;
		Map<Integer, List<List<String>>> treeLevels = new HashMap<Integer, List<List<String>>>();
		for (int i = 0; i < array.length(); i++) {
			treeLevels.put(i, breadthFirstSearch(array.getJSONObject(i), maxDepth));
		}

		for (int i = 0; i < treeLevels.size() - 1; i++) {
			List<List<String>> list0 = treeLevels.get(i);
			List<List<String>> list1 = treeLevels.get(i + 1);
			if (!list0.equals(list1))
				reduce = false;
		}

		if (reduce)
			return (new JSONArray()).put(array.getJSONObject(0));
		return array;
	}
}
