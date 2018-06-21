/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AP;

/**
 *
 * @author ANSH
 */

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.servlet.annotation.WebServlet;

import org.json.JSONException;
import twitter4j.TwitterException;

@WebServlet(name = "parser", urlPatterns = { "/parser" })
public class parse extends HttpServlet {

	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {

		try {

			String sentence = req.getParameter("text");
			String level = req.getParameter("level");
			String twt = req.getParameter("tweet");

			if (sentence != null) {
				resp.getWriter().append((new simple()).parse(sentence, Integer.valueOf(level)).toString());
			} else if (twt != null) {
				System.out.println(twt);
				// resp.getWriter().append(new
				// twitter2.Twitter2().toJson("@TampaPD").toString());
				resp.getWriter().append(new twitter2.Twitter2().toJson(twt).toString());
			} else {
				throw new IllegalArgumentException("'method' parameter required, must be 'methodA' or 'methodB' !");
			}

		} catch (JSONException e) {
			throw new RuntimeException(e);
		} catch (TwitterException ex) {
			Logger.getLogger(parse.class.getName()).log(Level.SEVERE, null, ex);
		}
	}
}
