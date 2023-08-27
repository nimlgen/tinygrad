import {
    BrowserRouter as Router,
    Switch,
    Route,
} from "react-router-dom";
import 'bootstrap/dist/css/bootstrap.min.css'

import history from './history'
import { PMain } from './pages/PMain';
import { PRun } from './pages/PRun';

import { createTheme, ThemeProvider } from '@mui/material/styles';
import { grey } from '@mui/material/colors';

import './css/main.css'

const theme = createTheme({
  palette: {
    primary: grey,
  },
  typography: {
    fontFamily: [
      '"Lucida Console", monospace',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
  },
});

function App() {
    return (
      <ThemeProvider theme={theme}>
        <Router history={history}>
            <Switch>
                {/* <Route
                    exact path="/product/:id"
                    render={(props) => <PProduct {...props} />}>
                </Route>
                <Route
                    path="/redirect"
                    render={(props) => <PRedirect {...props} />}>
                </Route>
                <Route
                    path="/search"
                    render={(props) => <PSearch {...props} />}>
                </Route>
                <Route
                    path="/brand/:id"
                    render={(props) => <PBrand {...props} />}>
                </Route>
                <Route
                    path="/compare"
                    render={(props) => <PCompare {...props} />}>
                </Route>
                <Route
                    path="/legal"
                    render={(props) => <PLegal {...props} />}>
                </Route>
                <Route
                    path="/info"
                    render={(props) => <PInfo {...props} />}>
                </Route>
                <Route
                    path="/help"
                    render={(props) => <PHelp {...props} />}>
                </Route> */}
                <Route
                    exact path="/"
                    render={(props) => <PMain {...props} />}>
                </Route>
                <Route
                    exact path="/run/:session_id"
                    render={(props) => <PRun {...props} />}>
                </Route>
                {/* <Route
                    exact path="/404"
                    render={(props) => <P404 {...props} />}>
                </Route>
                <Route
                    exact path="/sitemap"
                    render={(props) => <PSitemap {...props} />}>
                </Route>
                <Route
                    exact path="*"
                    unknownPage
                    render={(props) => <P404 {...props} />}>
                </Route> */}
            </Switch>
        </Router>
        </ThemeProvider>
    )
}

export default App