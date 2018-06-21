/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package twitter2;

import java.util.*;
import twitter4j.Twitter;
import twitter4j.TwitterFactory;

import twitter4j.conf.*;
import org.json.*;
import twitter4j.Status;
import twitter4j.TwitterException;

/**
 *
 * @author ANSH
 */
public class Twitter2 {

	public JSONObject toJson(String tweet) throws JSONException, TwitterException {

		ConfigurationBuilder cb = new ConfigurationBuilder();

		cb.setOAuthConsumerKey("39R2kqdzvVI9gOZUplYwzcz0X")
		  .setOAuthConsumerSecret("fn9n7YL5rfMChJccPaAJiZsFnM3QUxkvwW9v5KyHnkKjAqJZR0")
		  .setOAuthAccessToken("1009636733552594944-QOSdGMjBDWoxICezhnB2yd1sptxD6m")
		  .setOAuthAccessTokenSecret("fmSTtzb70yZFM88ptASKkhGxcUYS3Cg3Qh8WqgtBBoG7k")
		  .setTweetModeExtended(true);

		Twitter twitter = new TwitterFactory(cb.build()).getInstance();

		List<Status> statusList = null;

		try {
			statusList = twitter.getUserTimeline(tweet);
		} catch (TwitterException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		JSONArray ar1 = new JSONArray();
		JSONArray ar2 = new JSONArray();

		JSONObject obj = new JSONObject();

		for (Status status : statusList) {

			System.out.println(status.getUser().getName() + ":" + status.getText());
			String s = String.valueOf(status.getId());
			ar1.put(s);
			obj.put("ID", ar1.toString());
			ar2.put(status.getText());
			obj.put("Tweet", ar2);

		}

		System.out.println(obj);

		return obj;
	}
}
